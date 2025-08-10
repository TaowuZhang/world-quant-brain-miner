FROM n8nio/n8n:latest

# Install additional dependencies for custom nodes
RUN npm install -g npm@latest

# Copy custom nodes
COPY custom-nodes /home/node/.n8n/custom

# Set permissions
RUN chown -R node:node /home/node/.n8n/custom

# Expose port
EXPOSE 5678

# Use the default n8n entrypoint
CMD ["n8n"] 