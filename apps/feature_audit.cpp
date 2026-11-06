// Controlled frontend audit. Reuses the estimator's actual feature functions.
// Stable IDs identify (frame, pyramid level, integer level pixel), never tracks.
#include "frontend.hpp"
#include "features/brief.hpp"
#include "features/matcher.hpp"
#include "features/multiscale.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
namespace fs=std::filesystem;
using namespace scratch;
std::string stableId(int frame,int level,float x,float y){std::ostringstream s;s<<'f'<<std::setw(6)<<std::setfill('0')<<frame<<"_L"<<level<<"_x"<<int(x)<<"_y"<<int(y);return s.str();}
std::string frameName(int frame){std::ostringstream s;s<<"frame_"<<std::setw(6)<<std::setfill('0')<<frame;return s.str();}
struct FrameAudit {RgbdFrame frame;features::ImagePyramid pyramid;std::vector<FrameFeatures> stages;};
FrameAudit exportFrame(const fs::path& repo,const fs::path& out,int frame){
 FrameAudit f;std::string error;
 if(!LoadRgbdFrame((repo/"data/synthetic_bunny").string(),frame,f.frame,&error))throw std::runtime_error(error);
 FrontendParams p;p.fast.threshold=10;p.fast.max_keypoints=0;
 for(auto selection:{features::SelectionParams{false,0},features::SelectionParams{true,0},features::SelectionParams{false,1400},features::SelectionParams{true,1400}}){p.selection=selection;f.stages.push_back(ExtractFeatures(f.frame.rgb,p));}
 f.pyramid=features::BuildPyramid(f.stages[0].gray,p.pyramid);
 const char* names[]={"00_fast_all","01_harris_all","02_fast_quotas","03_harris_quotas"};
 auto id=[&](const FrameFeatures& a,int i){return stableId(frame,a.level[i],a.level_x[i],a.level_y[i]);};
 std::map<std::string,int> fastRank,harrisRank;
 for(size_t i=0;i<f.stages[0].keypoints.size();++i)fastRank[id(f.stages[0],i)]=i;
 for(size_t i=0;i<f.stages[1].keypoints.size();++i)harrisRank[id(f.stages[1],i)]=i;
 std::set<std::string> fastKept,harrisKept;
 for(size_t i=0;i<f.stages[2].keypoints.size();++i)fastKept.insert(id(f.stages[2],i));
 for(size_t i=0;i<f.stages[3].keypoints.size();++i)harrisKept.insert(id(f.stages[3],i));
 for(int stage=0;stage<4;++stage){const auto& a=f.stages[stage];std::ofstream csv(out/(frameName(frame)+"_"+names[stage]+".csv"));csv<<"stable_id,index,level,x,y,level_x,level_y,fast_score,harris_response,fast_order,harris_order,kept_fast_quota,kept_harris_quota,descriptor_hex\n"<<std::setprecision(12);
 for(size_t i=0;i<a.keypoints.size();++i){auto uid=id(a,i);csv<<uid<<','<<i<<','<<a.level[i]<<','<<a.keypoints[i].x<<','<<a.keypoints[i].y<<','<<a.level_x[i]<<','<<a.level_y[i]<<','<<a.keypoints[i].score<<','<<features::HarrisResponse(f.pyramid.levels[a.level[i]],int(a.level_x[i]),int(a.level_y[i]))<<','<<fastRank.at(uid)<<','<<harrisRank.at(uid)<<','<<fastKept.count(uid)<<','<<harrisKept.count(uid)<<',';for(auto word:a.descriptors[i].bits)csv<<std::hex<<std::setw(16)<<std::setfill('0')<<word;csv<<std::dec<<'\n';}
 if(!csv)throw std::runtime_error("Cannot write audit stage");}
 double scale=f.pyramid.scale_x.size()>1?f.pyramid.scale_x[1]:1.2;auto quotas=features::LevelQuotas(1400,f.pyramid.levels.size(),scale);
 std::ofstream levels(out/(frameName(frame)+"_levels.csv"));levels<<"level,quota,fast_all,harris_all,fast_quota,harris_quota\n";for(size_t l=0;l<quotas.size();++l)levels<<l<<','<<quotas[l]<<','<<f.stages[0].per_level_count[l]<<','<<f.stages[1].per_level_count[l]<<','<<f.stages[2].per_level_count[l]<<','<<f.stages[3].per_level_count[l]<<'\n';
 return f;
}
struct Decision{int best=-1,d1=257,d2=257;bool ratio=false;};
std::vector<Decision> decisions(const std::vector<features::BriefDescriptor>& a,const std::vector<features::BriefDescriptor>& b){std::vector<Decision> out(a.size());for(size_t i=0;i<a.size();++i){auto& v=out[i];for(size_t j=0;j<b.size();++j){int d=features::HammingDistance(a[i],b[j]);if(d<v.d1){v.d2=v.d1;v.d1=d;v.best=j;}else if(d<v.d2)v.d2=d;}v.ratio=b.size()>=2&&v.d1<.87*v.d2;}return out;}
int main(int argc,char**argv){try{
 fs::path repo=fs::current_path();int i=0,j=1;
 for(int a=1;a<argc;++a){std::string arg=argv[a];if(a+1>=argc)throw std::runtime_error("Expected --repo DIR --i N --j N");if(arg=="--repo")repo=argv[++a];else if(arg=="--i")i=std::stoi(argv[++a]);else if(arg=="--j")j=std::stoi(argv[++a]);else throw std::runtime_error("Unknown audit option: "+arg);}
 if(i<0||j<0||i==j)throw std::runtime_error("Choose two different nonnegative frame indices");
 repo=fs::absolute(repo);fs::path out=repo/"results"/"reports"/"figures"/"frontend"/("pair_"+std::to_string(i)+"_"+std::to_string(j));fs::create_directories(out);
 auto a=exportFrame(repo,out,i),b=exportFrame(repo,out,j);auto& A=a.stages[3];auto& B=b.stages[3];if(A.descriptors.size()<2||B.descriptors.size()<2)throw std::runtime_error("Not enough descriptors to audit ratios");
 auto ab=decisions(A.descriptors,B.descriptors),ba=decisions(B.descriptors,A.descriptors);auto accepted=features::MatchMutualRatio(A.descriptors,B.descriptors,.87);std::set<std::pair<int,int>> expected;for(auto m:accepted)expected.insert({m.query,m.train});
 std::ofstream csv(out/"matching_decisions.csv");csv<<"query,train,query_id,train_id,u_i,v_i,u_j,v_j,d1,d2,ratio_forward,reverse_d1,reverse_d2,ratio_reverse,mutual_best,accepted\n"<<std::setprecision(12);
 for(size_t q=0;q<ab.size();++q){auto f=ab[q];auto rev=ba[f.best];bool mutual=rev.best==int(q),keep=f.ratio&&rev.ratio&&mutual;if(keep!=bool(expected.count({q,f.best})))throw std::runtime_error("Audit diverged from actual matcher");
 csv<<q<<','<<f.best<<','<<stableId(i,A.level[q],A.level_x[q],A.level_y[q])<<','<<stableId(j,B.level[f.best],B.level_x[f.best],B.level_y[f.best])<<','<<A.keypoints[q].x<<','<<A.keypoints[q].y<<','<<B.keypoints[f.best].x<<','<<B.keypoints[f.best].y<<','<<f.d1<<','<<f.d2<<','<<f.ratio<<','<<rev.d1<<','<<rev.d2<<','<<rev.ratio<<','<<mutual<<','<<keep<<'\n';}
 std::ofstream config(out/"config.txt");config<<"frames="<<i<<','<<j<<"\nFAST threshold=10; levels=8; scale=1.2; budget=1400; ratio=0.87\nStage 0 and 1 have no cap: Harris alone changes order, not membership.\nStage 2 and 3 use the same quotas to isolate Harris selection.\nMatches are descriptor-only; depth and geometric checks occur later.\nStable IDs are local to a frame and pyramid level; they are not persistent physical tracks.\n";
 std::cout<<"AUDIT PASS: exact frontend and matcher checks; "<<A.keypoints.size()<<'/'<<B.keypoints.size()<<" selected keypoints, "<<accepted.size()<<" mutual matches. Output: "<<out<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
