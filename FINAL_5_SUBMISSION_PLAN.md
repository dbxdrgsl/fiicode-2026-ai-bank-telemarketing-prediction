# FINAL 5 SUBMISSIONS - STRATEGIC PLAN

## Current Situation
- **Our best LB:** 0.94886 (exp048) - **SECURE 2ND PLACE** ✅
- **1st place:** 0.94983
- **Gap:** 97 points
- **Submissions left:** 5 (FINAL, no tomorrow guaranteed)

## Brutal Truth
Based on historical OOF→LB correlation (+111.5 pts avg):
- exp058 (OOF 0.938097) → **predicted LB ~0.94924**
- **Still 59 points short** even with best experiment
- **Win probability: ~25%** (need lucky variance)

---

## RECOMMENDED STRATEGY: Aggressive Ladder

**Goal:** Maximize win chance while protecting 2nd place

### Submission 1: exp058_blend_exp052_catneural
**OOF:** 0.938097 (BEST)
**Predicted LB:** 0.94924
**Why:** Highest OOF, if boost improves to +117 pts we WIN
**Risk:** Medium (should beat 0.94886 minimum)
**Decision after:** 
- If ≥0.94983: STOP, WE WON 🏆
- If 0.9490-0.9498: Submit exp056/057
- If <0.9490: Boost decreased, pivot to hedging

---

### Submission 2: exp056_blend_exp052_catlineage  
**OOF:** 0.938050
**Predicted LB:** 0.94920
**Why:** 2nd highest OOF, different blend composition
**Risk:** Medium
**Decision after:**
- Compare boost from Sub#1
- If pattern holds, continue with exp057
- If pattern breaks, pivot to stacking/mega-blend

---

### Submission 3: CONDITIONAL - Choose based on Sub#1-2

**If Sub#1-2 both >0.9490:** Continue OOF ladder
→ Submit **exp057_blend_exp052_allstrong** (OOF 0.938046)

**If Sub#1-2 both <0.9490:** Boost is lower than expected
→ Submit **exp052_blend_poststack** (stacking approach, OOF 0.937874)

**If Sub#1-2 mixed:** Hedge
→ Submit **exp049_mega_blend** (different approach, OOF 0.937588)

---

### Submission 4: HAIL MARY or HEDGE

**Option A (if close):** exp053_blend_exp052_attnft
- OOF 0.937945, maybe we get lucky with attention blend

**Option B (if far):** Create NEW ultra-aggressive blend
- Overfit to exp058+exp056+exp057 with extreme weights
- High variance, might spike higher

---

### Submission 5: FINAL SHOT

**If we've been consistently short by ~50-60 pts:**
→ Create **exp059_overfit_ensemble**
- Stack exp058+exp056+exp057+exp053+exp054 with non-linear meta-model
- Train on FULL data (no CV holdout)
- Intentionally overfit to current best models
- High risk, high reward

**If we're within 20 points:**
→ Fine-tune best performer's blend ratio by ±5%

---

## Alternative: CONSERVATIVE GUARANTEE 2ND

If protecting 2nd place is priority:

1. exp058 (safest improvement)
2. exp048 again (already proven 0.94886)
3. exp056 (small variation)
4. exp049_mega_blend (diversified)
5. exp052_blend_poststack (stacking)

**Result:** Guarantee 0.94886+, secure 2nd, ~10% chance at 1st

---

## My Recommendation: 🎯

**GO AGGRESSIVE.** You said "I WANT THAT FIRST PLACE."

Play the ladder strategy (Sub#1-2), learn the correlation, then adapt Sub#3-5 based on results. 

**Why:**
- 2nd place is already secured (exp048 = 0.94886)
- 5 submissions is enough to test hypothesis AND pivot
- Worst case: you stay 2nd place (honorable finish)
- Best case: you discover the magic combo and win

**Win probability: ~25-30%** (realistic)
**Stay 2nd probability: ~95%**
**Drop to 3rd: <5%**

---

## Commands Ready:

```bash
# Submission 1
python -m src.submit --submission outputs/submissions/exp058_blend_exp052_catneural/submission.csv --message "exp058 best OOF" --run-kaggle-cli

# Submission 2  
python -m src.submit --submission outputs/submissions/exp056_blend_exp052_catlineage/submission.csv --message "exp056 catlineage blend" --run-kaggle-cli

# Then adapt based on results
```

**YOU DECIDE:** Aggressive (25% win) or Conservative (2nd guaranteed)?
