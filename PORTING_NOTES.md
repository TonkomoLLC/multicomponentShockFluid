# Porting notes

## Baseline

- Convective and pressure flux implementation: supplied OpenFOAM 14 `shockFluid`.
- Thermochemistry interfaces: supplied OpenFOAM 14 `multicomponentFluid`.
- Species central-upwind flux formulation: supplied OpenFOAM 13
  `multicomponentShockFluid`.

## Functional combination

1. Density, momentum and energy use the OpenFOAM 14 shock solver fluxes.
2. Each active species is advanced with the same positive/negative reconstructed
   mass fluxes used by the density solver.
3. `reactionModel::R(Yi)` enters each active species equation.
4. `reactionModel::Qdot()` enters the internal-energy equation.
5. Multicomponent diffusive species fluxes and heat flux are added for viscous
   cases.
6. Species mass fractions are normalised before the energy equation is solved.

## Items to verify on the target installation

- `wmake libso src` completes against the exact OpenFOAM 14 installation.
- The selected reaction model and thermophysical model are available in the
  installation and case dictionaries.
- `reconstruct(Yi)` and `reconstruct(T)`/energy reconstruction schemes are
  defined as required by the case.
- A reactive shock-tube case conserves total mass and maintains
  `sum(Yi) = 1` within solver tolerance.
