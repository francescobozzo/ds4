{
  description = "DwarfStar 4 - DeepSeek V4 Flash inference engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.rocmSupport = builtins.elem system [
            "x86_64-linux"
            "aarch64-linux"
          ];
        };

        commonNativeInputs = with pkgs; [
          gnumake
          coreutils
          bison
          which
          gcc
        ];

        rocmPkgs = pkgs.rocmPackages;
        rocmArch = "gfx1151";

        rocmCFlags = builtins.concatStringsSep " " [
          "-O3"
          "-ffast-math"
          "-g"
          "-fno-finite-math-only"
          "-pthread"
          "-D__HIP_PLATFORM_AMD__"
          "-Wno-unused-command-line-argument"
          "--offload-arch=${rocmArch}"
          "-I${rocmPkgs.rocwmma}/include"
          "-I${rocmPkgs.hipcub}/include"
          "-I${rocmPkgs.hipblas}/include"
          "-I${rocmPkgs.hipblaslt}/include"
          "-I${rocmPkgs.hipblas-common}/include"
          "-I${rocmPkgs.rocprim}/include"
        ];

        rocmLdlibs = builtins.concatStringsSep " " [
          "-lm"
          "-pthread"
          "-L${rocmPkgs.hipblas}/lib"
          "-L${rocmPkgs.hipblaslt}/lib"
          "-lhipblas"
          "-lhipblaslt"
        ];

        rocmLibraryPath = pkgs.lib.makeLibraryPath [
          rocmPkgs.hipblas
          rocmPkgs.hipblaslt
          rocmPkgs.clr
        ];
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "ds4-rocm";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = commonNativeInputs ++ [
            rocmPkgs.clr
          ];

          buildInputs = [
            rocmPkgs.clr
            rocmPkgs.hipblas
            rocmPkgs.hipblas-common
            rocmPkgs.hipblaslt
            rocmPkgs.rocwmma
            rocmPkgs.hipcub
            rocmPkgs.rocprim
          ];

          ROCM_PATH = "${rocmPkgs.clr}";
          ROCM_ARCH = rocmArch;
          GPU_BACKEND = "rocm";
          CC = "${pkgs.gcc}/bin/gcc";
          HIP_PATH = "${rocmPkgs.clr}";
          HIP_PLATFORM = "amd";
          HIP_CLANG_PATH = "${rocmPkgs.clr}/bin";
          HIPCC = "${rocmPkgs.clr}/bin/hipcc";
          NATIVE_CPU_FLAG = "-march=x86-64";
          LD_LIBRARY_PATH = rocmLibraryPath;

          buildPhase = ''
            make rocm -j$NIX_BUILD_CORES ROCM_ARCH="$ROCM_ARCH" \
              CC="$CC" \
              HIPCC="$HIPCC" \
              GPU_BACKEND="$GPU_BACKEND" \
              ROCM_PATH="$ROCM_PATH" \
              NATIVE_CPU_FLAG="$NATIVE_CPU_FLAG" \
              ROCM_LDLIBS="${rocmLdlibs}" \
              ROCM_CFLAGS="${rocmCFlags}"
          '';

          installPhase = ''
            mkdir -p $out/bin
            for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
              [ -x "$bin" ] && install -m 755 "$bin" "$out/bin/"
            done
          '';

          meta = {
            description = "DwarfStar 4 (ROCm ${rocmArch})";
            homepage = "https://github.com/antirez/ds4";
            license = pkgs.lib.licenses.mit;
            platforms = pkgs.lib.platforms.linux;
          };
        };

        devShells.default = pkgs.mkShell {
          name = "ds4-dev";

          buildInputs = commonNativeInputs ++ [
            rocmPkgs.clr
            rocmPkgs.hipblas
            rocmPkgs.hipblas-common
            rocmPkgs.hipblaslt
            rocmPkgs.rocwmma
            rocmPkgs.hipcub
            rocmPkgs.rocprim
            rocmPkgs.rocprofiler-sdk
          ];

          ROCM_PATH = "${rocmPkgs.clr}";
          ROCM_ARCH = rocmArch;
          ROCM_CFLAGS = rocmCFlags;
          ROCM_LDLIBS = rocmLdlibs;
          GPU_BACKEND = "rocm";
          CC = "${pkgs.gcc}/bin/gcc";
          HIP_PATH = "${rocmPkgs.clr}";
          HIP_PLATFORM = "amd";
          HIP_CLANG_PATH = "${rocmPkgs.clr}/bin";
          HIPCC = "${rocmPkgs.clr}/bin/hipcc";
          NATIVE_CPU_FLAG = "-march=x86-64";
          LD_LIBRARY_PATH = rocmLibraryPath;
        };
      }
    );
}
