FROM node:18-alpine
WORKDIR /app

# Copy package.json and package-lock.json to install dependencies
COPY package*.json ./

RUN npm install

EXPOSE 3000

CMD ["npm", "start"]
