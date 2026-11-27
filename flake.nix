{
  description = "kmslap: Kennesaw Motorsports lap sim (FSAE EV)";

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
          # this project, installed editable (so `kmslap` is a command, with completion)
          # pointing at the checkout: code edits apply without rebuilding the env
          editable = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };
          editableBuild = final: prev: {
            python-data-analysis = prev.python-data-analysis.overrideAttrs (old: {
              # only what hatchling reads to build the editable wheel
              src = lib.fileset.toSource {
                root = old.src;
                fileset = lib.fileset.unions [
                  (old.src + "/pyproject.toml")
                  (old.src + "/kmslap/__init__.py")
                ];
              };
              nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
            });
          };
          pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
            (lib.composeManyExtensions [
              pyproject-build-systems.overlays.default overlay wheelLibs editable editableBuild
            ]);
        in
        pythonSet.mkVirtualEnv "python-data-analysis-env" workspace.deps.all;
    in {
      # tab completion for the `kmslap` command, without touching shell rc files:
      # `nix profile install .#completions` puts it where zsh (via NIX_PROFILES'
      # site-functions) and bash-completion look. It only runs `kmslap` on TAB, so
      # it's inert outside this repo's shell. Regenerate completions/ with
      # `kmslap --show-completion zsh|bash` if the command's name changes.
      packages = forAll (pkgs: {
        completions = pkgs.runCommand "kmslap-completions" { } ''
          install -Dm644 ${./completions/_kmslap} $out/share/zsh/site-functions/_kmslap
          install -Dm644 ${./completions/kmslap.bash} $out/share/bash-completion/completions/kmslap
        '';
      });

      devShells = forAll (pkgs: {
        default = let venv = pythonEnv pkgs; in pkgs.mkShell {
          # rerun: the viewer for kmslap/reporting/viz.py, first on PATH so it wins over the
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
            # the editable install of this project resolves to here
            export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
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
