FROM nvidia/cuda:12.3.2-devel-ubi8
ENV DEBIAN_FRONTEND=noninteractive

# Disable subscription-manager
# Add workaround to disable subscription-manager plugin
#RUN echo -e "[main]\nenabled=0" > /etc/dnf/plugins/subscription-manager.conf && \
#    echo -e "enabled=0" > /etc/yum/pluginconf.d/subscription-manager.conf


# Install dependencies
RUN yum update -y && \
    yum install -y wget gcc make cmake zlib-devel openssl-devel libXext libXrender libSM libffi-devel  \
    git bzip2 bzip2-devel readline sqlite-devel ncurses-devel xz xz-devel libGL

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

# Install Python 3.10.9 with required libraries
RUN pyenv install 3.10.9
RUN pyenv global 3.10.9

# Install pip3 and upgrade it
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN ~/.pyenv/shims/python3 get-pip.py
RUN rm get-pip.py
RUN pyenv rehash

# Add pyenv and pip to PATH
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc

# Continue with the rest of your Dockerfile
WORKDIR /app
COPY requirements.txt /app

RUN ~/.pyenv/shims/pip install --upgrade pip setuptools wheel
RUN ~/.pyenv/shims/pip install numpy==1.24.2
RUN ~/.pyenv/shims/pip install dlib==19.24.0
RUN ~/.pyenv/shims/pip install -r requirements.txt

RUN git clone https://github.com/NVlabs/SPADE.git

WORKDIR /app/SPADE
RUN ~/.pyenv/shims/pip install -r requirements.txt

WORKDIR /app
ADD . /app
ENV HOST=${HOST:-0.0.0.0}
ENV PORT=${PORT:-8000}
EXPOSE ${PORT}
# to print Python output to console
ENV PYTHONUNBUFFERED=1
CMD ["~/.pyenv/shims/python3", "manage.py", "runserver", "0.0.0.0:8000"]
