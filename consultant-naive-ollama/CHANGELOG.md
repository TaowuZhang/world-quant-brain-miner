# Changelog

## [2.0.0] - 2025-08-10

### 🚀 Major Features Added

#### Web Dashboard
- **Real-time Monitoring**: Comprehensive web dashboard with Flask
- **Status Monitoring**: GPU, Ollama, Orchestrator, and WorldQuant status
- **Manual Controls**: Trigger alpha generation, mining, and submission
- **Real-time Logs**: Filtered alpha generator logs and system logs
- **Auto-refresh**: 30-second automatic updates
- **Responsive Design**: Modern UI with gradient backgrounds

#### Alpha Orchestrator
- **Integrated Workflow**: Single orchestrator managing all components
- **Scheduling**: Intelligent scheduling for mining and submission
- **Daily Rate Limiting**: Ensures compliance with WorldQuant limits
- **Continuous Mode**: Automated 24/7 operation
- **Error Handling**: Robust error handling and recovery

#### Docker Integration
- **GPU Support**: Full NVIDIA GPU acceleration
- **Multi-service**: Docker Compose with multiple services
- **Persistent Storage**: Volume mounts for results and logs
- **Health Checks**: Automated system monitoring

### 🔧 Technical Improvements

#### Alpha Generation
- **Ollama Integration**: Replaced Kimi with local Ollama
- **Model Support**: llama3.2:3b and llama2:7b models
- **GPU Acceleration**: CUDA support for faster inference
- **Batch Processing**: Configurable batch sizes

#### Alpha Mining
- **Expression Mining**: Automated parameter variation mining
- **Pattern Recognition**: Identifies successful alpha patterns
- **Optimization**: Suggests improvements to existing alphas

#### Alpha Submission
- **Daily Limits**: Respects WorldQuant submission limits
- **Success Filtering**: Only submits high-performing alphas
- **Rate Limiting**: Prevents API rate limit issues

### 📁 File Structure

#### Core Files
- `alpha_generator_ollama.py` - Main alpha generation script
- `alpha_orchestrator.py` - Workflow orchestration
- `alpha_expression_miner.py` - Alpha expression mining
- `successful_alpha_submitter.py` - Alpha submission
- `web_dashboard.py` - Flask web dashboard
- `templates/dashboard.html` - Dashboard HTML template

#### Docker Files
- `Dockerfile` - Container image definition
- `docker-compose.gpu.yml` - GPU-enabled deployment
- `docker-compose.yml` - CPU-only deployment
- `.dockerignore` - Docker ignore rules

#### Configuration
- `requirements.txt` - Python dependencies
- `credential.txt` - WorldQuant credentials
- `start_gpu.bat` - Windows GPU startup script
- `start_dashboard.bat` - Windows dashboard startup script

#### Documentation
- `README.md` - Main project documentation
- `README_Docker.md` - Docker-specific documentation
- `CHANGELOG.md` - This changelog

### 🗑️ Cleaned Up Files

#### Removed (No Longer Needed)
- `alpha_generator.py` - Replaced by Ollama version
- `promising_alpha_miner.py` - Functionality integrated into orchestrator
- `alpha_polisher.py` - Functionality integrated into miner
- `alpha_101_testing.py` - Replaced by integrated testing
- `test_dashboard.py` - Replaced by web dashboard
- `clean_up_logs.py` - Functionality in dashboard
- `start.sh` - Replaced by Docker
- `start.bat` - Replaced by new scripts
- `start_orchestrator.bat` - Integrated into main scripts
- `credential.example.txt` - No longer needed

### 🔄 Workflow Changes

#### Before (v1.0)
- Manual Kimi interface
- Separate scripts for each component
- No automation or scheduling
- Limited monitoring capabilities

#### After (v2.0)
- Automated Ollama integration
- Integrated orchestrator workflow
- Continuous operation with scheduling
- Comprehensive web dashboard monitoring
- Docker containerization
- GPU acceleration

### 📊 Performance Improvements

#### Generation Speed
- **Before**: ~10-15 seconds per alpha (Kimi API)
- **After**: ~3-5 seconds per alpha (Local Ollama + GPU)

#### Monitoring
- **Before**: Manual log checking
- **After**: Real-time web dashboard with auto-refresh

#### Automation
- **Before**: Manual intervention required
- **After**: Fully automated 24/7 operation

### 🛠️ Technical Stack

#### Backend
- **Python 3.8**: Main application language
- **Flask**: Web dashboard framework
- **Requests**: HTTP client for APIs
- **Schedule**: Task scheduling
- **PyTorch**: GPU acceleration support

#### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **NVIDIA CUDA**: GPU acceleration
- **Ollama**: Local LLM serving

#### Frontend
- **HTML5/CSS3**: Dashboard interface
- **JavaScript**: Real-time updates
- **Responsive Design**: Mobile-friendly layout

### 🔒 Security Improvements

- **Local Processing**: All LLM inference happens locally
- **Credential Protection**: Secure credential storage
- **Network Isolation**: Docker network isolation
- **API Rate Limiting**: Respects external API limits

### 📈 Monitoring & Logging

- **Real-time Metrics**: GPU usage, generation rates, success rates
- **Log Filtering**: Alpha-specific log filtering
- **Error Tracking**: Comprehensive error monitoring
- **Activity Timeline**: Recent activity tracking

---

## [1.0.0] - 2025-08-09

### Initial Release
- Basic alpha generation with Kimi interface
- Manual alpha testing and submission
- Simple script-based workflow
- No automation or monitoring

---

**Note**: This changelog documents the major evolution from a simple script-based system to a comprehensive, automated, containerized alpha generation platform with real-time monitoring and GPU acceleration.
