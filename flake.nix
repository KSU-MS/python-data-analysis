{
  description = "FSAE performance analysis / accel sim";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Python deps come from uv.lock (pyproject.toml), built by uv2nix, so Nix and
    # non-Nix setups (`uv sync`) get the exact same versions.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      # prefer prebuilt wheels from the lock; Nix patches them to run on NixOS
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      pythonEnv = pkgs:
        let
          python = pkgs.python313;
          # native libs some wheels link against that aren't bundled in the wheel
          wheelLibs = final: prev: {
            rerun-sdk = prev.rerun-sdk.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.udev ];
            });
            # plugins for commercial solvers (knitro, snopt, mosek, MATLAB, ...) link
            # libraries we don't have; casadi only loads a plugin when it's asked for
            casadi = prev.casadi.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;
            });
          };
          pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
            (lib.composeManyExtensions [ pyproject-build-systems.overlays.default overlay wheelLibs ]);
        in
        pythonSet.mkVirtualEnv "python-data-analysis-env" workspace.deps.all;
    in {
      devShells = forAll (pkgs: {
        default = let venv = pythonEnv pkgs; in pkgs.mkShell {
          # rerun: the viewer for accel/viz.py, first on PATH so it wins over the
          # wheel's bundled viewer (which needs GL/Wayland libs NixOS doesn't expose).
          # pyproject.toml pins rerun-sdk to this version.
          packages = [ pkgs.rerun venv pkgs.uv ];
          env = {
            # the env is built by Nix from uv.lock: `uv run` uses it as-is, and
            # never syncs or downloads anything
            UV_NO_SYNC = "1";
            UV_PYTHON_DOWNLOADS = "never";
            UV_PROJECT_ENVIRONMENT = "${venv}";
          };
          shellHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
            export UV_PYTHON="${venv}/bin/python"
            # strip notebook outputs on commit (see .gitattributes). Uses the
            # nbstripout on PATH rather than a /nix/store path that could get GC'd.
            if git rev-parse --git-dir >/dev/null 2>&1; then
              git config filter.nbstripout.clean nbstripout
              git config filter.nbstripout.smudge cat
              git config filter.nbstripout.required true
              git config diff.ipynb.textconv "nbstripout -t"
            fi
          '';
        };
        # ARM cross toolchain for `make -C controller teensy` (~1 GB, so kept out
        # of the default shell)
        firmware = pkgs.mkShell {
          packages = [ pkgs.gcc-arm-embedded pkgs.gnumake ];
        };
      });
    };
}
