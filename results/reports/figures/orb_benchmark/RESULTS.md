# ORB benchmark results

OpenCV ORB is compared with the production project ORB-style extractor. Both use the same project Hamming matcher and GN ICP backend; this is not ORB-SLAM.

| Method | Solved pairs | Mean features | GT precision | ATE (mm) | Extraction (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Project ORB-style | 35/35 | 1013.9 | 97.00% | 1.688 | 81.345 |
| OpenCV ORB | 35/35 | 993.9 | 97.07% | 1.692 | 4.405 |

See protocol.txt for all controls, timing definitions, remaining implementation differences, and failure handling.
