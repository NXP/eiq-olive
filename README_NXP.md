# NXP Olive AI
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/Olive?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/Olive/)
[![Latest version](https://img.shields.io/badge/nxp--olive--ai-0.1.0-brightgreen)](https://nl4-nxrm.sw.nxp.com/repository/EIQ-pypi/simple/nxp-olive-ai/)

## Development

Default branch used in this project is `eiqtlk-dev`. It should be updated via `main` branch, that points
to used stable revision of Olive framework.

## Release process

Every push to `eiqtlk-dev` branch is processed by CI producing Python wheel with version `0.0.0.dev0`.
This version can be considered as daily build.

To publish new version of this library, change latest version defined on top of this Readme
and push git tag with new version in format `x.y.z`.
