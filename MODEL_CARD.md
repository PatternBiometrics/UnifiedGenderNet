# UnifiedGenderNet – Model Card

## Task
Binary gender prediction from **hand** or **face** images using a single model (UMCC or MAG).  

## Training data
- Derived from [HaGRID](https://github.com/hukenovs/hagrid) (HaGRID-Derived Face–Hand Pairs) (15,064 aligned face–hand pairs)
- Subject-exclusive splits (train/val/test)  
- Labels: **apparent gender**  

## Protocol
- Input: 224×224 RGB  
- Optimizer: AdamW, cosine LR, batch 32  
- Loss: BCE (UMCC), BCE+modality CE (MAG)  
- Thresholds: 0.50 global, subgroup thresholds only for fairness  

## Results 
  - Details will be provided after paper publication

**Baseline (EffNetV2-S, UMCC, Full aug):**
- Test Accuracy ≈ 91%  
- Balanced Accuracy ≈ 91%  
- ROC–AUC ≈ 0.97  
- PR–AUC ≈ 0.97  

📌 Full results (ablations, fairness, cross-dataset) will be added post-publication

## Known limitations
- Lower stability on hands than faces  
- Cross-dataset drops (FairFace, 11K Hands)  
- Fairness disparities remain even after calibration ( more robuste approach is needed)

## License
- Code: MIT  
- Data: derived HaGRID, redistributed under same license (see : [LICENSE](https://github.com/PatternBiometrics/UnifiedGenderNet/blob/main/HAGRID_DERIVED_LICENSE.pdf))
