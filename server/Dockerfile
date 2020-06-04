FROM node:latest
WORKDIR /usr/src/app

COPY ./package*.json ./

RUN apt update
RUN apt install python3-pip
RUN pip3 install -r requirements
RUN npm install

COPY ./ .

EXPOSE 3000
CMD ["npm", "start"]

