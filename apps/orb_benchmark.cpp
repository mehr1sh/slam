// Controlled extraction comparison: project ORB-style vs OpenCV ORB.
// Both use the SAME project Hamming matcher, depth gates, RANSAC and GN ICP.
#include "features/matcher.hpp"
#include "tracking/rgbd_icp.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <sstream>
#include <stdexcept>
using namespace rgbd_icp;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
struct Method {
  std::string name;
  std::vector<ImageFeatures> features;
  std::vector<double> extract_ms;
  std::vector<Pose> trajectory{Pose{}};
  Pose velocity;
  double matches=0, valid=0, ransac_inliers=0, gt_correct=0, match_ms=0, registration_ms=0;
  double rotation_error=0, translation_error=0;
  int solved=0;
};
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t m=v.size()/2;
  return v.size()%2 ? v[m] : (v[m-1]+v[m])*.5;
}
ImageFeatures openCVFeatures(const cv::Mat &image, const Config &c) {
  // Same project BT.601 gray conversion as the production scratch frontend.
  auto g=features::GrayFromInterleaved(image.data,image.cols,image.rows,image.step,image.channels());
  cv::Mat gray(g.height,g.width,CV_8UC1,g.pixels.data());
  auto orb=cv::ORB::create(c.features,1.2f,8,16,0,2,cv::ORB::HARRIS_SCORE,31,c.fast_threshold);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat desc;
  orb->detectAndCompute(gray,cv::noArray(),keypoints,desc);
  ImageFeatures out;
  for (int i=0;i<desc.rows;++i) {
    out.pixels.emplace_back(keypoints[i].pt.x,keypoints[i].pt.y);
    out.levels.push_back(keypoints[i].octave);
    out.angles_rad.push_back(keypoints[i].angle * float(std::acos(-1.) / 180.));
    features::BriefDescriptor d;
    // Explicit byte->bit conversion is portable and preserves Hamming distance.
    for (int bit=0;bit<256;++bit)
      if (desc.at<uint8_t>(i,bit/8)&(1u<<(bit%8))) d.set(bit);
    out.descriptors.push_back(d);
  }
  return out;
}
void writeKeypoints(const fs::path &out, int frame, const std::string &method,
                    const ImageFeatures &features) {
  if(features.levels.size()!=features.pixels.size() ||
     features.angles_rad.size()!=features.pixels.size())
    throw std::runtime_error("Incomplete feature audit metadata");
  fs::create_directories(out / "keypoints");
  std::ostringstream name;
  name << "frame_" << std::setw(6) << std::setfill('0') << frame << '_' << method << ".csv";
  std::ofstream f(out / "keypoints" / name.str());
  f << "index,x,y,level,angle_rad\n" << std::setprecision(12);
  for(size_t i=0;i<features.pixels.size();++i)
    f << i << ',' << features.pixels[i].x() << ',' << features.pixels[i].y()
      << ',' << features.levels[i] << ',' << features.angles_rad[i] << '\n';
  if(!f) throw std::runtime_error("Cannot write keypoints");
}
std::vector<Pose> readGT(const std::string &dir) {
  std::ifstream f(dir+"/groundtruth.txt");
  std::vector<Pose> out;
  std::string line;
  while (std::getline(f,line)) {
    if(line.empty()||line[0]=='#') continue;
    std::istringstream row(line);
    double stamp,x,y,z,qx,qy,qz,qw;
    if(!(row>>stamp>>x>>y>>z>>qx>>qy>>qz>>qw)) throw std::runtime_error("Malformed groundtruth");
    Eigen::Quaterniond q(qw,qx,qy,qz);
    if(q.norm()<1e-10) throw std::runtime_error("Invalid groundtruth quaternion");
    out.push_back({q.normalized().toRotationMatrix(),V3(x,y,z)});
  }
  return out;
}
void writePose(std::ostream &f,int i,const Pose &p) {
  Eigen::Quaterniond q(p.R);
  f<<i<<' '<<p.t.transpose()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
}
int main(int argc,char **argv) {
  try {
    std::string dir=argc>1?argv[1]:"data/synthetic_bunny";
    fs::path out=argc>2?argv[2]:"outputs/orb_benchmark";
    int repeats=3,limit=0;
    for(int a=3;a<argc;++a) {
      std::string flag=argv[a];
      if((flag=="--repeats"||flag=="--frames")&&a+1<argc) {
        int value=std::stoi(argv[++a]);
        if(flag=="--repeats") repeats=value; else limit=value;
      } else throw std::runtime_error("Usage: orb_benchmark DATASET OUTPUT [--repeats N] [--frames N]");
    }
    if(repeats<1||limit<0) throw std::runtime_error("Invalid repeats/frames");
    cv::setNumThreads(1); cv::setRNGSeed(42);
    Config c; Camera k=readCamera(dir);
    std::vector<Frame> frames;
    for(int i=0;limit==0||i<limit;++i) {
      std::ostringstream filename;filename<<dir<<'/'<<std::setw(6)<<std::setfill('0')<<i<<".png";
      if(!fs::exists(filename.str())) break;
      frames.push_back(readFrame(dir,i,k));
    }
    if(frames.size()<2) throw std::runtime_error("Need >=2 frames");
    fs::create_directories(out); fs::remove(out/"run_complete.txt");
    std::array<Method,2> methods;
    methods[0].name="project_orb_style";methods[1].name="opencv_orb";
    std::ofstream perframe(out/"frames.csv"),pairs(out/"pairs.csv");
    perframe<<"method,frame,keypoints,extraction_median_ms\n"<<std::setprecision(12);
    pairs<<"method,pair,status,matches,valid_3d,ransac_inliers,gt_correct_3d,gt_precision_3d,"
           "matching_median_ms,registration_ms,rotation_error_deg,translation_error_m\n"<<std::setprecision(12);
    for(int i=0;i<int(frames.size());++i) {
      // Alternating method order limits systematic warm-cache/order bias.
      for(int order=0;order<2;++order) {
        int m=(order+i)%2;auto &method=methods[m];
        auto extract=[&](){return m==0?extractFeatures(frames[i].rgb,c):openCVFeatures(frames[i].rgb,c);};
        extract(); // warmup, excluded from timing
        std::vector<double> times;ImageFeatures f;
        for(int r=0;r<repeats;++r) {
          auto start=Clock::now();f=extract();
          times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        }
        // Export outside the timed section. Coordinates are the actual result
        // used by the registration backend, not a second extraction.
        writeKeypoints(out,i,method.name,f);
        method.extract_ms.push_back(median(times));
        perframe<<method.name<<','<<i<<','<<f.pixels.size()<<','<<median(times)<<'\n';
        method.features.push_back(std::move(f));
      }
    }
    // Estimate all poses BEFORE reading any ground truth.
    struct Row {int i;PairResult result;double match_ms,registration_ms;};
    std::array<std::vector<Row>,2> rows;
    for(int i=0;i<int(frames.size())-1;++i) for(int order=0;order<2;++order) {
      int m=(order+i)%2;auto &method=methods[m];
      const auto &fa=method.features[i],&fb=method.features[i+1];
      auto match=[&](){return features::MatchMutualRatio(fa.descriptors,fb.descriptors,c.ratio);};
      match();std::vector<double> times;
      for(int r=0;r<repeats;++r) {
        auto start=Clock::now();auto matches=match();
        times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        if(matches.size()>fa.pixels.size()) throw std::runtime_error("Invalid match count");
      }
      auto start=Clock::now();
      auto result=registerFramesWithFeatures(frames[i],frames[i+1],k,c,fa,fb,c.seed+i);
      double registration_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
      Pose step=result.accepted?result.plane:method.velocity;
      if(result.accepted) method.velocity=step;
      method.trajectory.push_back(method.trajectory.back()*step);
      rows[m].push_back({i,result,median(times),registration_ms});
      std::cout<<method.name<<' '<<i<<"->"<<i+1<<": "<<result.status<<" matches="<<result.matches<<'\n';
    }
    bool hasGT=fs::exists(dir+"/groundtruth.txt");
    auto gt=hasGT?readGT(dir):std::vector<Pose>{};
    if(hasGT&&gt.size()<frames.size()) throw std::runtime_error("Too few GT rows");
    std::ofstream summary(out/"summary.csv");
    summary<<"method,frames,pairs,solved_pairs,failed_pairs,mean_keypoints,mean_matches,mean_valid_3d,"
             "mean_ransac_inliers,gt_precision_3d,median_extraction_ms,mean_matching_ms,mean_registration_ms,"
             "ATE_RMSE_m,mean_accepted_rotation_error_deg,mean_accepted_translation_error_m\n"<<std::setprecision(12);
    for(int m=0;m<2;++m) {
      auto &method=methods[m];
      for(auto &row:rows[m]) {
        int i=row.i;auto &r=row.result;
        double rot=0,trans=0;int correct=0;
        if(hasGT) {
          Pose truth=gt[i].inverse()*gt[i+1];
          std::vector<V3> target,source;int match_count;
          matchedFeaturePoints(frames[i],frames[i+1],k,c,method.features[i],method.features[i+1],target,source,match_count);
          for(size_t j=0;j<source.size();++j)
            if((truth.apply(source[j])-target[j]).norm()<c.ransac_threshold) ++correct;
          if(r.accepted) {
            rot=angleDeg(r.plane.R*truth.R.transpose());trans=(r.plane.t-truth.t).norm();
            method.rotation_error+=rot;method.translation_error+=trans;
          }
        }
        method.solved+=r.accepted;method.matches+=r.matches;method.valid+=r.valid3d;
        method.ransac_inliers+=r.inliers;method.gt_correct+=correct;
        method.match_ms+=row.match_ms;method.registration_ms+=row.registration_ms;
        pairs<<method.name<<','<<i<<','<<r.status<<','<<r.matches<<','<<r.valid3d<<','<<r.inliers<<',';
        if(hasGT) pairs<<correct;
        pairs<<',';
        if(hasGT&&r.valid3d) pairs<<double(correct)/r.valid3d;
        pairs<<','<<row.match_ms<<','<<row.registration_ms<<',';
        if(hasGT&&r.accepted) pairs<<rot;
        pairs<<',';if(hasGT&&r.accepted) pairs<<trans;pairs<<'\n';
      }
      double ate=0,keypoints=0;
      std::ofstream traj(out/(method.name+"_trajectory.txt"));
      traj<<"# frame tx ty tz qx qy qz qw (frame-zero world; no GT used in estimation)\n"<<std::setprecision(15);
      for(size_t i=0;i<frames.size();++i) {
        keypoints+=method.features[i].pixels.size();writePose(traj,int(i),method.trajectory[i]);
        if(hasGT) ate+=((gt[0]*method.trajectory[i]).t-gt[i].t).squaredNorm();
      }
      double n=double(rows[m].size());
      summary<<method.name<<','<<frames.size()<<','<<int(n)<<','<<method.solved<<','<<int(n)-method.solved<<','
             <<keypoints/frames.size()<<','<<method.matches/n<<','<<method.valid/n<<','<<method.ransac_inliers/n<<',';
      if(hasGT&&method.valid) summary<<method.gt_correct/method.valid;
      summary<<','<<median(method.extract_ms)<<','<<method.match_ms/n<<','<<method.registration_ms/n<<',';
      if(hasGT) summary<<std::sqrt(ate/frames.size());
      summary<<',';
      if(hasGT&&method.solved) summary<<method.rotation_error/method.solved;
      summary<<',';if(hasGT&&method.solved) summary<<method.translation_error/method.solved;summary<<'\n';
    }
    std::ofstream protocol(out/"protocol.txt");
    protocol<<"OpenCV "<<CV_VERSION<<"; C++ "<<__cplusplus<<"; cv_threads=1; seed=42\n"
      "Extraction: one warmup then median of "<<repeats<<" runs per frame; includes BT.601 gray conversion, detector creation and descriptor adapter.\n"
      "Matching: one warmup then median of "<<repeats<<" runs; identical project mutual Hamming ratio matcher.\n"
      "Method order alternates per frame/pair. Image I/O excluded. Registration timed once; includes matching, depth lifting, RANSAC and both GN stages.\n"
      "Both: requested budget=1400, scale=1.2, levels=8, FAST=10, Harris ranking, border=16, 256 bits; actual counts reported.\n"
      "Ratio=0.87 bidirectional + mutual; 3x3 depth edge=0.012m; RANSAC threshold=0.009m, 650 iterations.\n"
      "Frontend comparison, not identical implementations or an ORB-SLAM benchmark. Different scores, smoothing, sampling and descriptor patterns remain.\n"
      "Same dense clouds, rejection, source(j)->target(i) convention, RANSAC seeds, point-GN and Huber plane-GN.\n"
      "GT is read only after all estimation; GT precision=pooled correct valid-depth pairs / valid-depth pairs, threshold=0.009m.\n"
      "Keypoint CSVs contain every retained position, pyramid level and orientation from the timed extraction. Export is untimed.\n"
      "No descriptor is compared across methods. No loop closure; ATE includes flagged constant-velocity fallbacks, frame-zero anchor, no fitted alignment/scale.\n"
      "Accepted relative error means exclude failed pairs; failures are separately reported. Timings are machine dependent.\n";
    perframe.close();pairs.close();summary.close();protocol.close();
    if(!perframe||!pairs||!summary||!protocol) throw std::runtime_error("Cannot write benchmark files");
    std::ofstream complete(out/"run_complete.txt");complete<<"frames="<<frames.size()<<"\n";
    if(!complete) throw std::runtime_error("Cannot write completion marker");
    std::cout<<"Benchmark outputs: "<<out<<'\n';return 0;
  } catch(const std::exception &e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
