# Multi-stage build optimized for a smaller production image

# 1. Install dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# 2. Build the Next.js app (standalone output)
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_BASE_PATH="/"
ENV NEXT_BASE_PATH=${NEXT_BASE_PATH}
ENV NEXT_PUBLIC_BASE_PATH=${NEXT_BASE_PATH}
RUN npm run build
RUN npm prune --production
RUN mkdir -p tmp/jobs

# 3. Production runner (distroless Node.js)
FROM gcr.io/distroless/nodejs20-debian12:nonroot AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ARG NEXT_BASE_PATH="/"
ENV NEXT_BASE_PATH=${NEXT_BASE_PATH}
ENV NEXT_PUBLIC_BASE_PATH=${NEXT_BASE_PATH}

# Copy standalone output and static assets
COPY --from=builder --chown=nonroot:nonroot /app/.next/standalone ./ 
COPY --from=builder --chown=nonroot:nonroot /app/.next/static ./.next/static
COPY --from=builder --chown=nonroot:nonroot /app/public ./public
COPY --from=builder --chown=nonroot:nonroot /app/tmp ./tmp

EXPOSE 3000
CMD ["/app/server.js"]
