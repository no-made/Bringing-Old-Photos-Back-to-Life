# Reinherit Toolkits: Old Photos' Restorer

This web app is part of the ReInHerit Toolkit and is abased on the paper **Bringing Old Photos Back to Life**. It allows you to restore old photos by removing scratches and enhancing the quality of the image. The app is based on the work of Ziyu Wan, Bo Zhang, Dongdong Chen, Pan Zhang, Dong Chen, Jing Liao, and Fang Wen. The paper was presented at the IEEE/CVF Conference on Computer Vision and Pattern Recognition in 2020.

<img src='static/assets/images/HR_result.png'>

## Table of Contents
1. [Overview](#overview)
2. [How to Run the App](#how-to-run-the-app)
    - [Prerequisites](#prerequisites)
    - [Requirements](#requirements)
3. [Installation](#installation)
    - [Clone the Repository and Set Up Models](#clone-the-repository-and-set-up-models)
    - [Manage Environment Variables and Secret Keys](#manage-environment-variables-and-secret-keys)
4. [Running the App](#running-the-app)
    - [Using Python Virtual Environment](#using-python-virtual-environment)
    - [Using Docker](#using-docker)
5. [Application Usage](#application-usage)
    - [Landing Page](#landing-page)
    - [Input Page](#input-page)
    - [Output Page](#output-page)
6. [Citation](#citation)
7. [License](#license)


## Overview
ReInHerit Toolkits: Old Photos' Restorer is a web application designed to restore old photos using state-of-the-art deep learning techniques. 
The application allows users to either upload their own images or select from a gallery of preloaded images. 
The process is efficient, and the output provides a comparison between the original and restored images. 

## How to run the app
You can run the app using one of two methods:
- By creating a Python virtual environment.
- By using Docker.

Follow the relevant instructions based on your chosen method.

### Prerequisites

- **Python 3.10**: Ensure Python 3.10 is installed on your machine. You can download it from the [official website](https://www.python.org/downloads/ ) or follow this [installation guide](https://realpython.com/installing-python/). 
- **Javascript Enabled**: Make sure JavaScript is enabled in your browser. Follow this [guide](https://www.enable-javascript.com/) if needed. 

### Requirements
Depending on the method chosen to run the app:
  - **Python Virtual Environment**: It's recommended to use Conda for managing virtual environments. Check if Conda is installed by running:
    ```
    conda --version 
    ``` 
    If not installed, follow the [Anaconda installation guide](https://docs.anaconda.com/anaconda/install/): 
  - **Docker**: Ensure Docker is installed and running on your operating system. If you’re new to Docker, refer to the[ official documentation](https://docs.docker.com/).


## Installation
### Clone the Repository and Set Up Models 
1. Clone this repository on your local machine
2. Download the Landmark Detection Pretrained Model:
    ```
    cd Face_Detection/
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    cd ../
    ```
3. Download the Pretrained Models:
   - Place the Face_Enhancement/checkpoints.zip under ./Face_Enhancement/.
   - Place the Global/checkpoints.zip under ./Global/.
   - Unzip them using:
    ```
    cd Face_Enhancement/
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
    unzip face_checkpoints.zip
    cd ../
    cd Global/
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
    unzip global_checkpoints.zip
    cd ../
    ```
### Manage Environment Variables and Secret Keys
#### ___1. Django secret key___: 
- Generate a Django secret key: 
   ```
   python getYourDjangoKey.py
   ```
- Copy the generated key and paste it into the **DJANGO_KEY** field of the **.env** file. 


#### ___2. Google Analytics key (optional)___
- Follow the steps [here](https://analytics.google.com/analytics/web/) to create a Google Analytics property and obtain the **GA_KEY**.
- Paste the **GA_KEY** into the **.env** file.

#### ___3. Browse or Gallery mode___
- Choose between gallery mode or browse mode by setting the MODE in the **.env** file:
  ```
  MODE=gallery  # or MODE=input
  ```

## Running the App
### Using Python Virtual Environment
1. Create a Virtual Environment and Install Requirements:
    - Navigate to the folder with requirements.txt:  
      ```
      conda create --name my_env_name python=3.10
      conda activate my_env_name
      pip install -r requirements.txt
      ``` 
    - Replace my_env_name with your desired environment name.
2. Run the Web App Locally:
    - Navigate to the folder containing manage.py and run:
      ```
      python manage.py runserver
      ```
3. Open the Home Page:
    - Open a browser and go to the address:  
      ```
      http://localhost:8000
      ```

### Using Docker

1. Build the Docker image:
   - In the root of the repository, run:
     ```
     docker build -t oldphoto .
     ```
2. Run the Docker Container:
   - In the root of the repository, run:
     ```
     docker run --env-file=.env -p 8000:8000 oldphoto
     ```
3. Open the Home Page:
   - Open a browser and go to the address: 
     ```
     http://localhost:8000
     ```

## Application Usage

### Landing Page
Click on the '**Start to restore**' button to begin the demo

### Input Page
- **Gallery Scenario**:
  - Select images from the gallery to restore.
  - Use the '**with scratches**' checkbox for damaged images. 
  - Click on the **PROCESS** button to start the restoration process.
- **Browse Scenario**: 
  - Click **BROWSE** button to upload your images.
  - Select images and use checkboxes for scratches or high DPI.
  - Click on the **PROCESS** button to start the restoration process.
  
Note: The processing time depends on the number of images you upload. The more images you upload, the longer it will take to process.

### Output Page
- View the original and restored images, along with a comparison.
- Download the restored images or restart the process.



## Citation

```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```


## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
