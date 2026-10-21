# Flatpak Packaging for Súlimo

This directory contains the Flatpak manifest and build scripts for packaging Súlimo into an isolated, reproducible Flatpak application and single-file bundle.

## Files
- `io.github.antux912012.Sulimo.json`: Application Flatpak manifest targeting `org.gnome.Platform//50` and `org.gnome.Sdk//50`.
- `build_flatpak.sh`: Automation script that builds the application into a local OSTree repository and exports `sulimo_<version>.flatpak`.

## Building the Flatpak Bundle
```bash
./packaging/flatpak/build_flatpak.sh
```

The resulting single-file bundle will be exported to the repository root:
```bash
flatpak install --user sulimo_0.3.0.flatpak
```

