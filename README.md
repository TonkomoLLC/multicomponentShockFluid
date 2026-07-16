# multicomponentShockFluid for OpenFOAM 14

`multicomponentShockFluid` combines the density-based central-upwind fluxes of
OpenFOAM 14 `shockFluid` with the species, reaction and multicomponent transport
framework used by OpenFOAM 14 `multicomponentFluid`.

## OpenFOAM 14 port

The port includes the OpenFOAM 14 API changes relevant to this module:

- derives from `basicFluidSolver`;
- constructs `U` with `dimensions::velocity`;
- uses `mesh.schemes().lookupOrDefault(...)`;
- uses `mesh.poly().topoChanged()` in the motion corrector;
- uses `reactionModel` and `libreactionModels`;
- uses the OpenFOAM 14 `psiMulticomponentThermo` and
  `fluidMulticomponentThermophysicalTransportModel` interfaces;
- uses the OpenFOAM 14 versions of the shock-specific patch fields.

The momentum-transport model is constructed even when the molecular viscosity
is zero. This is required because the OpenFOAM 14 `reactionModel` constructor
requires a momentum-transport-model reference. Viscous stress, species
diffusion and heat conduction are still omitted when the case is identified as
inviscid.

## Build

Load the OpenFOAM 14 environment, then run:

```bash
./Allwmake
```

The library is written to:

```text
$FOAM_USER_LIBBIN/libmulticomponentShockFluid.so
```

## Run-time selection

Use the module with `foamRun` or `foamMultiRun`:

```text
solver          multicomponentShockFluid;
```

Load the user library in `system/controlDict` when it is not loaded globally:

```text
libs
(
    "libmulticomponentShockFluid.so"
);
```

The thermodynamic model must be compatible with `psiMulticomponentThermo`, and
shock reconstruction entries must be provided for the solved fields, including
`reconstruct(Yi)` for species.

## Validation status

The source was ported by direct comparison with the supplied OpenFOAM 14
`shockFluid` and `multicomponentFluid` modules. An OpenFOAM 14 installation was
not available in the artifact-generation environment, so compilation and case
execution were not performed here.
