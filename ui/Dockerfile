FROM node:24-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

FROM nginx:1.29-alpine

# The nginx image renders templates at startup, substituting only variables that are set,
# so nginx's own $host/$uri stay intact. The default keeps docker compose working.
ENV BACKEND_URL=http://backend:8000
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
