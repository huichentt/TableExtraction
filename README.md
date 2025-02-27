# Table Extraction

This repository provides a table understanding API for scientific literature. It can extract tables, figures, and their captions from a given PDF of scientific literature, and then extract the table structure along with its content. This table structure recognition part was inspired by the [unitable](https://github.com/poloclub/unitable).

## Quickstart

To run the table extraction API:

Run *build.sh* to build the Docker container based on the Dockerfile.
Then, run the container
```
docker run --gpus all -d --name tablefigure -p 8000:8000 -e NVIDIA_DISABLE_REQUIRE=true tableextraction:tf /bin/bash start.sh
```

Please contact the repository owner if you need the pre-trained or fine-tuned models.