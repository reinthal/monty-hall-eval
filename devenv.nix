{
  pkgs,
  lib,
  ...
}: {
  env.PYTHONPATH = ".";
  env.SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";

  packages = with pkgs; [
    git
    pre-commit
  ];

  languages = {
    python = {
      uv.enable = true;
    };
  };

  enterShell = ''
    echo "which uv: $(which uv)"
    echo " version: $(uv --version)"
  '';
}
