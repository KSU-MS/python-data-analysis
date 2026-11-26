{
  description = "FSAE performance analysis / accel sim";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      devShells = forAll (pkgs:
        let
          # Use nixpkgs-built packages (prebuilt in the binary cache) instead of
          # pip/uv wheels, which need nix-ld to find native libs on NixOS.
          python = pkgs.python3.withPackages (ps: with ps; [
            numpy
            pandas
            scipy
            matplotlib
            plotly
            seaborn
            numba
            requests
            pyserial
            cantools
            tkinter
            ipykernel
            ipympl
            jupyterlab
            pytest
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ python pkgs.ruff pkgs.nbstripout ];
            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
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
