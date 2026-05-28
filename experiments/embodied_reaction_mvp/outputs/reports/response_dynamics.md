# response_ready Dynamics Diagnostic

Verifies that `response_ready` timing is driven by internal path dynamics
rather than a fixed timer. All values derived from simulation logs.

## Cross-scenario summary

| Scenario                               | n          | mean IEI   | std IEI    | ttp@≥0.99  | r(rec→IEI) | r(sil→IEI) | r(ttp→depth) |
|----------------------------------------|------------|------------|------------|------------|------------|------------|--------------|
| speech_then_silence                    | 7          | 1.98       | 0.16       | 0.0%       | +0.65      | -0.52      | +1.00        |
| repeated_speech_silence                | 5          | 5.07       | 1.88       | 0.0%       | -0.97      | +0.61      | +1.00        |
| sudden_approach_while_speaking         | 5          | 1.97       | 0.05       | 0.0%       | +0.95      | +0.93      | +1.00        |
| random_ambient_noise                   | 8          | 3.08       | 0.74       | 0.0%       | +0.12      | +0.37      | +1.00        |
| long_listening                         | 12         | 2.47       | 1.35       | 0.0%       | -0.13      | +0.54      | +1.00        |

---

## speech_then_silence

**7 firing(s) total.**

### 1. response_ready intervals
- List: 1.93, 2.18, 2.17, 1.85, 1.82, 1.92 s
- Mean: 1.978 s
- Std:  0.159 s
- Min:  1.817 s   Max: 2.183 s

### 2. ttp saturation
- Steps with ttp ≥ 0.99: **0.00%**
- Max continuous saturation: **0.00 s**

### 3. ttp at response_ready firing
- Values: 0.668, 0.720, 0.799, 0.801, 0.723, 0.732, 0.769
- Mean: 0.744   Std: 0.048
- Min:  0.668   Max: 0.801

### 4. ttp immediately after discharge
- Post-fire values: 0.163, 0.153, 0.137, 0.137, 0.153, 0.151, 0.144
- Mean: 0.148   Std: 0.010
- Min:  0.137   Max: 0.163
- Discharge depth (ttp_before − ttp_after): 0.505, 0.567, 0.662, 0.664, 0.570, 0.581, 0.626
  Mean: 0.596   Std: 0.058

### 5. recovery at response_ready firing
- Values: 0.076, 0.202, 0.227, 0.127, 0.141, 0.204, 0.240
- Mean: 0.174   Std: 0.060

### 6. Correlation checks
- recovery at firing → next interval:      +0.65 — moderate positive
- silence_duration at firing → next interval: -0.52 — moderate negative
- ttp at firing → discharge depth:         +1.00 — strong positive

### 7. Interpretation
- **Clock-like?** No — std=0.159 s indicates genuine interval variation.
- **ttp rebuilding dynamically?** Yes — ttp never saturates (0.0% at ≥0.99). Post-discharge values vary (0.137-0.163, std=0.010), so rebuild length differs each cycle.
- **Recovery modulating interval?** Yes — r=+0.65 between recovery-at-fire and next interval. Higher recovery → longer next gap, as designed.

---

## repeated_speech_silence

**5 firing(s) total.**

### 1. response_ready intervals
- List: 6.13, 5.47, 6.37, 2.30 s
- Mean: 5.067 s
- Std:  1.883 s
- Min:  2.300 s   Max: 6.367 s

### 2. ttp saturation
- Steps with ttp ≥ 0.99: **0.00%**
- Max continuous saturation: **0.00 s**

### 3. ttp at response_ready firing
- Values: 0.674, 0.671, 0.668, 0.703, 0.809
- Mean: 0.705   Std: 0.060
- Min:  0.668   Max: 0.809

### 4. ttp immediately after discharge
- Post-fire values: 0.162, 0.163, 0.163, 0.156, 0.135
- Mean: 0.156   Std: 0.012
- Min:  0.135   Max: 0.163
- Discharge depth (ttp_before − ttp_after): 0.512, 0.508, 0.504, 0.547, 0.674
  Mean: 0.549   Std: 0.072

### 5. recovery at response_ready firing
- Values: 0.087, 0.070, 0.071, 0.224, 0.168
- Mean: 0.124   Std: 0.069

### 6. Correlation checks
- recovery at firing → next interval:      -0.97 — strong negative
- silence_duration at firing → next interval: +0.61 — moderate positive
- ttp at firing → discharge depth:         +1.00 — strong positive

### 7. Interpretation
- **Clock-like?** No — std=1.883 s indicates genuine interval variation.
- **ttp rebuilding dynamically?** Yes — ttp never saturates (0.0% at ≥0.99). Post-discharge values vary (0.135-0.163, std=0.012), so rebuild length differs each cycle.
- **Recovery modulating interval?** Yes — r=-0.97 between recovery-at-fire and next interval. Higher recovery → shorter next gap, as designed.

---

## sudden_approach_while_speaking

**5 firing(s) total.**

### 1. response_ready intervals
- List: 1.92, 1.93, 2.02, 2.02 s
- Mean: 1.971 s
- Std:  0.053 s
- Min:  1.917 s   Max: 2.017 s

### 2. ttp saturation
- Steps with ttp ≥ 0.99: **0.00%**
- Max continuous saturation: **0.00 s**

### 3. ttp at response_ready firing
- Values: 0.931, 0.678, 0.736, 0.767, 0.781
- Mean: 0.779   Std: 0.094
- Min:  0.678   Max: 0.931

### 4. ttp immediately after discharge
- Post-fire values: 0.111, 0.162, 0.150, 0.144, 0.141
- Mean: 0.141   Std: 0.019
- Min:  0.111   Max: 0.162
- Discharge depth (ttp_before − ttp_after): 0.820, 0.516, 0.587, 0.623, 0.640
  Mean: 0.637   Std: 0.113

### 5. recovery at response_ready firing
- Values: 0.051, 0.118, 0.179, 0.206, 0.248
- Mean: 0.160   Std: 0.077

### 6. Correlation checks
- recovery at firing → next interval:      +0.95 — strong positive
- silence_duration at firing → next interval: +0.93 — strong positive
- ttp at firing → discharge depth:         +1.00 — strong positive

### 7. Interpretation
- **Clock-like?** YES — std=0.053 s is below 0.08 s. Intervals are still nearly uniform.
- **ttp rebuilding dynamically?** Yes — ttp never saturates (0.0% at ≥0.99). Post-discharge values vary (0.111-0.162, std=0.019), so rebuild length differs each cycle.
- **Recovery modulating interval?** Yes — r=+0.95 between recovery-at-fire and next interval. Higher recovery → longer next gap, as designed.
- **Synchronized?** This scenario still looks too synchronized (std=0.053 s). Consider checking whether the freeze path or recovery pulse is resetting all paths to a common state, or whether the base refire gap is the binding constraint.

---

## random_ambient_noise

**8 firing(s) total.**

### 1. response_ready intervals
- List: 3.47, 2.05, 3.00, 2.12, 3.70, 3.30, 3.92 s
- Mean: 3.079 s
- Std:  0.739 s
- Min:  2.050 s   Max: 3.917 s

### 2. ttp saturation
- Steps with ttp ≥ 0.99: **0.00%**
- Max continuous saturation: **0.00 s**

### 3. ttp at response_ready firing
- Values: 0.664, 0.677, 0.738, 0.689, 0.762, 0.678, 0.689, 0.703
- Mean: 0.700   Std: 0.034
- Min:  0.664   Max: 0.762

### 4. ttp immediately after discharge
- Post-fire values: 0.164, 0.161, 0.149, 0.159, 0.144, 0.161, 0.159, 0.156
- Mean: 0.157   Std: 0.007
- Min:  0.144   Max: 0.164
- Discharge depth (ttp_before − ttp_after): 0.499, 0.515, 0.589, 0.529, 0.618, 0.517, 0.530, 0.546
  Mean: 0.543   Std: 0.040

### 5. recovery at response_ready firing
- Values: 0.050, 0.110, 0.162, 0.146, 0.186, 0.107, 0.156, 0.229
- Mean: 0.144   Std: 0.054

### 6. Correlation checks
- recovery at firing → next interval:      +0.12 — negligible
- silence_duration at firing → next interval: +0.37 — weak positive
- ttp at firing → discharge depth:         +1.00 — strong positive

### 7. Interpretation
- **Clock-like?** No — std=0.739 s indicates genuine interval variation.
- **ttp rebuilding dynamically?** Yes — ttp never saturates (0.0% at ≥0.99). Post-discharge values vary (0.144-0.164, std=0.007), so rebuild length differs each cycle.
- **Recovery modulating interval?** Weak — r=+0.12. Recovery influence on next interval is not strong enough to be the dominant source of variation.

---

## long_listening

**12 firing(s) total.**

### 1. response_ready intervals
- List: 1.87, 2.20, 2.18, 1.83, 1.97, 1.97, 2.03, 6.50, 2.12, 2.22, 2.23 s
- Mean: 2.465 s
- Std:  1.346 s
- Min:  1.833 s   Max: 6.500 s

### 2. ttp saturation
- Steps with ttp ≥ 0.99: **0.00%**
- Max continuous saturation: **0.00 s**

### 3. ttp at response_ready firing
- Values: 0.664, 0.702, 0.807, 0.804, 0.707, 0.785, 0.778, 0.799, 0.686, 0.765, 0.799, 0.819
- Mean: 0.760   Std: 0.054
- Min:  0.664   Max: 0.819

### 4. ttp immediately after discharge
- Post-fire values: 0.164, 0.156, 0.136, 0.137, 0.156, 0.140, 0.142, 0.138, 0.160, 0.144, 0.137, 0.134
- Mean: 0.145   Std: 0.011
- Min:  0.134   Max: 0.164
- Discharge depth (ttp_before − ttp_after): 0.499, 0.546, 0.672, 0.667, 0.551, 0.645, 0.637, 0.661, 0.526, 0.622, 0.662, 0.685
  Mean: 0.614   Std: 0.065

### 5. recovery at response_ready firing
- Values: 0.050, 0.208, 0.230, 0.126, 0.198, 0.228, 0.282, 0.152, 0.145, 0.217, 0.248, 0.124
- Mean: 0.184   Std: 0.065

### 6. Correlation checks
- recovery at firing → next interval:      -0.13 — negligible
- silence_duration at firing → next interval: +0.54 — moderate positive
- ttp at firing → discharge depth:         +1.00 — strong positive

### 7. Interpretation
- **Clock-like?** No — std=1.346 s indicates genuine interval variation.
- **ttp rebuilding dynamically?** Yes — ttp never saturates (0.0% at ≥0.99). Post-discharge values vary (0.134-0.164, std=0.011), so rebuild length differs each cycle.
- **Recovery modulating interval?** Weak — r=-0.13. Recovery influence on next interval is not strong enough to be the dominant source of variation.

---
