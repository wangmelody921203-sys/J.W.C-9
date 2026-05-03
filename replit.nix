{ pkgs }: {
  deps = [
    pkgs.php83
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.numpy
    pkgs.python312Packages.opencv4
    pkgs.python312Packages.onnxruntime
  ];
}
