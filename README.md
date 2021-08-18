# Prophet Forecast

Attempt to use the Prophet system to forecast natural gas usage for the following day.  Pulling data from proprietary Microsoft Access databases, which are not included in this project.

## Description

Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels.

Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS.

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps - so let's get started!

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

## Getting Started

### Dependencies

* Docker
* Anaconda
* Streamlit
* Facebook Prophet

### Installing

* Install Docker Desktop
* Create a Docker Container
```
docker run --name anaconda-prophet -it -d --cap-add SYS_ADMIN --cap-add DAC_READ_SEARCH -p 8051:8501 continuumio/anaconda3

      --name                                                        [ The name of the container.  If not specific, docker comes up with one randomly. ]
      -it                                                                  [ (i) Keep STDIN open even if not attached (t) Allocate a pseudo-tty : For dummies: (creates an interactive process/shell) ]
      -d                                                                  [ Detached ]
      --cap-add SYS_ADMIN
      --cap-add DAC_READ_SEARCH       [ These 2 commands allow mounting inside of a docker container ]
      -p                                                                  [ The ports to use (internal:external) ]
```
* Create a conda virtual environment and activate it
```
Run these commands from the CLI in the container

    conda create --name python36-env python=3.6
    conda activate python36-env
```
* Install some necessary packages
```
 conda install Cython
    conda install pystan -c conda-forge
    conda install fbprophet -c conda-forge
    conda install pyodbc
    pip install plotly
    pip install streamlit
    pip install pandas_access
    conda install -c anaconda sqlite
```
* If using Visual Studio Code, connect visual studio code   (make sure extension "Remote - Containers" is installed)
```
a) Open visual studio code
  b) Select Remote Explorer on tool bar
  c) choose appropriate container, right-click and select attach to container
     (if it doesn't show, you may need to open the command palette and choose Remote-Containers: Open Folder in Container...  \\wsl$\docker-desktop)
  d) select /home directory
```
* If using Access database (or other file-based Database server), set up share to allow Access database communication
```
apt-get update && apt-get install -y cifs-utils vim
mkdir /mnt/arminius
       echo '//{IP_ADDRESS}/Data /mnt/arminius cifs iocharset=utf8,credentials=/root/.smbcredentials 0 0' > /etc/fstab
       -OR-
       echo '//{IP_ADDRESS}/Data /mnt/arminius cifs iocharset=utf8,credentials=/root/.smbcredentials,nobrl 0 0' > /etc/fstab
      -OR-
       echo '//{IP_ADDRESS}/Company/Database /mnt/arminius cifs iocharset=utf8,credentials=/root/.smbcredentials,nobrl 0 0' > /etc/fstab

vim /root/.smbcredentials

username=USERNAME
password=PASSWORD

mount -a
```

### Executing program

* Run streamlit server
```
cd /home
streamlit run app.py
```

* Startup procedure
```
Run docker desktop
Start container anaconda-prophet
Open Visual Studio Code
Open project C:\development\projects\anaconda-prophet
Remote Explorer > Dev Containers > continuumio/anaconda3 (/anaconda-prophet) > /home > Open Container
Go to Terminal at bottom
conda activate python36-env
mount -a
streamlit run app.py
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Tyson Manwarren

## Version History


* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [facebook prophet](https://facebook.github.io/prophet/)

