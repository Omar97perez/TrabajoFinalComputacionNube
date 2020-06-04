FROM node:latest
WORKDIR /usr/src/app

COPY ./package*.json ./

RUN pip install -r requirements
RUN npm install

COPY ./ .

EXPOSE 3000
CMD ["npm", "start"]

