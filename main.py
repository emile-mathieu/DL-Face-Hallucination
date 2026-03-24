# Model architecture: As the paper: 
# Low-res (variable size like 20–50)
#         ↓
# Resize → 48×48 (Iin)
#         ↓
# Model
#         ↓
# Output → 100×100 (SR)
#         ↓
# Compare with HR target (IH) → compute loss