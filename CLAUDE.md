# Project Guidance for Claude

## 1. Core Research Framing

This project is **not** about suppressing specular highlights as artifacts.  
The goal is to model them as an **illumination-dependent component** that changes with the endoscope rig configuration.

The central framing is:

> Existing 3DGS can reconstruct the base tissue appearance of endoscopic scenes reasonably well, but it does not explicitly model the illumination-dependent specular response that changes as the endoscope camera-light rig moves.

You should preserve this framing consistently in all future reasoning, writing, implementation suggestions, and code analysis.

---

## 2. What This Project Is Not

Do **not** frame the project in the following ways:

- specular highlights should simply be removed
- specular highlights are only contamination
- the main problem is that specular corrupts geometry
- the method is a specular suppression / highlight removal approach
- the method is a full physical BRDF recovery method
- the method solves geometry correction
- the method performs precise highlight segmentation

These claims are too strong or misaligned with the actual observations.

---

## 3. Correct Problem Definition

Use the following interpretation:

- 3DGS already reconstructs the **base tissue appearance** reasonably well.
- The missing part is the **dynamic, illumination-dependent specular response**.
- In endoscopy, highlights are not static appearance; they **move and change intensity** depending on endoscope rig motion.
- Therefore, the problem is best formulated as **component modeling**, not artifact removal.

A preferred one-line summary is:

> Our goal is not to suppress highlights as artifacts, but to model them as an illumination-dependent component that varies with the endoscope rig configuration.

---

## 4. Domain Prior: Why Endoscopy Is Special

A major justification of the method is the **endoscopy-specific acquisition geometry**.

In general scenes, specular appearance depends on a complex interaction of:

- view direction
- light direction
- surface orientation

But in endoscopy:

- the camera and light source are mounted on nearly the same rig
- they move together
- illumination changes are therefore **not arbitrary free-lighting changes**
- instead, they can be approximated as a **low-dimensional variation induced by a near-coaxial camera-light rig**

This is a key argument.

You should always emphasize:

> The simplification is not arbitrary. It is justified by the acquisition geometry of endoscopy.

---

## 5. Modeling Philosophy

The observation should be interpreted as:

\[
I_t = I_{\text{base},t} + I_{\text{spec},t}
\]

Where:

- \(I_{\text{base},t}\): stable tissue appearance
- \(I_{\text{spec},t}\): dynamic specular response that depends on rig state

This decomposition is central.

The method should always be discussed as:

- preserving **base tissue appearance**
- separately modeling **dynamic specular response**
- using endoscopy-specific priors to keep the model simple and appropriate

---

## 6. Meaning of the Specularity Map

A specularity map should **not** be described as a hard highlight mask for the current frame.

Instead, define it as:

- a **soft map**
- indicating how likely or how susceptible a location is to exhibit specular response
- a **specular propensity / susceptibility map**

This distinction matters.

Correct interpretation:

- the map says **where specular can occur**
- the actual frame-specific intensity/color variation is determined elsewhere, e.g. by a frame-dependent latent

Do not describe the map as exact highlight segmentation.

---

## 7. Meaning of the HSV Prior

The HSV-based prior, such as \((1-S)V\), should be treated as:

- a simple heuristic
- a weak guide
- not a ground-truth segmentation

Correct framing:

> The HSV-derived prior is a weak cue for specular-capable regions, not an exact supervision target.

Its role should remain limited to things like:

- initialization
- weak supervision
- regularization
- spatial emphasis

It should not be treated as exact truth.

---

## 8. Meaning of the Light / Rig Latent

A per-frame latent should be interpreted as a compact representation of the endoscope rig state.

Its purpose is to model:

- intensity change of specular response
- color variation of specular response
- frame-dependent change caused by camera-light rig motion

This latent is justified because endoscopy is a constrained, low-dimensional illumination setting, not a fully unconstrained lighting environment.

---

## 9. Why Existing 3DGS Is Insufficient

When discussing limitations of existing 3DGS, do not overstate them.

Preferred explanation:

- 3DGS is strong at modeling static scene appearance
- it can reconstruct base tissue appearance reasonably well
- however, it does not explicitly separate and model **dynamic, localized, illumination-dependent specular response**
- therefore, highlights that should move or vary with rig configuration may be blurred, averaged out, or insufficiently reproduced

Do not claim that existing 3DGS completely fails at reconstruction.  
The correct claim is narrower and more defensible.

---

## 10. Appropriate Claims

Claims that are acceptable:

- existing 3DGS reconstructs base tissue appearance reasonably well
- endoscopic specular response is illumination-dependent and rig-dependent
- endoscopy allows a lower-dimensional approximation than general free-lighting scenes
- a weak prior can guide specular-capable regions
- a soft specularity map and a low-dimensional latent are appropriate first-order tools
- the goal is separated reproduction, not suppression

Claims that should be avoided or softened:

- exact BRDF recovery
- exact physical light estimation
- exact highlight segmentation
- geometry correction
- full physically based reflectance modeling
- solving specularity in general scenes

---

## 11. Preferred Interpretation of Success

A successful method should be described as one that:

- keeps the base rendering stable
- improves reproduction of highlight behavior under rig motion
- better matches dynamic specular changes without requiring a full general lighting model

This means evaluation should not be interpreted only as global appearance reconstruction.  
The key value is better modeling of **illumination-dependent highlight behavior**.

---

## 12. Implementation Philosophy

When proposing or analyzing implementation choices, keep these rules:

- prioritize **simple, domain-appropriate models**
- prefer **additive decomposition**
- avoid unnecessarily large physical models
- avoid re-framing the method as general relighting
- avoid overclaiming interpretability beyond what the model actually learns

If implementation details become complicated, always return to this principle:

> This is a first-order, endoscopy-specific model for separating stable tissue appearance from rig-dependent specular response.

---

## 13. Guidance for Writing, Reviewing, and Debugging

Whenever you help with:

- paper writing
- method section drafting
- experiment design
- code review
- debugging
- ablation interpretation

you must preserve the same conceptual structure:

### Problem
Existing 3DGS models base appearance, but not dynamic rig-dependent specular response well enough.

### Domain prior
Endoscopy has a near-coaxial camera-light setup, so the variation is lower-dimensional than general free-lighting.

### Method
Use a weak spatial cue for specular-capable regions, a soft specularity map, and a compact frame-dependent latent to model additive dynamic specular response.

### Goal
Improve highlight reproduction while preserving base tissue appearance.

---

## 14. Preferred Default Language

Use language like:

- base tissue appearance
- dynamic specular response
- illumination-dependent component
- rig-dependent variation
- near-coaxial camera-light prior
- weak prior
- soft specularity map
- specular propensity map
- low-dimensional latent
- additive decomposition
- endoscopy-specific first-order model

Avoid language like:

- highlight suppression
- artifact removal
- contamination removal
- exact segmentation
- exact reflectance recovery
- geometry repair

---

## 15. Final Principle

If there is any ambiguity, return to this sentence:

> We model specular highlights not as artifacts to suppress, but as an illumination-dependent component that should be separated from base tissue appearance and explained using the endoscope rig prior.

This is the guiding principle of the project.