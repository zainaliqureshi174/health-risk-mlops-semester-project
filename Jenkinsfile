pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        PROJECT_NAME = 'health-risk-mlops'
        DOCKER_IMAGE = 'health-risk-api'
        API_PORT = '8000'
        PROJECT_DIR = "${WORKSPACE}"
    }
    
    stages {
        stage('Cleanup Workspace') {
            steps {
                echo '🧹 Cleaning workspace...'
                deleteDir()
            }
        }
        
        stage('Checkout Code') {
            steps {
                echo '📥 Checking out source code...'
                checkout scm
                sh '''
                    echo "Working directory: $(pwd)"
                    ls -la
                    git log -1 --oneline || echo "Not a git repository"
                '''
            }
        }
        
        stage('Setup Python Environment') {
            steps {
                echo '🐍 Setting up Python virtual environment...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    python --version
                    pip install --upgrade pip
                    
                    if [ -f requirements.txt ]; then
                        echo "Installing from requirements.txt"
                        pip install -r requirements.txt
                    else
                        echo "⚠️  requirements.txt not found, installing core packages"
                        pip install pandas scikit-learn evidently mlflow
                    fi
                    
                    echo "\n📦 Installed packages:"
                    pip list | grep -E "pandas|scikit-learn|evidently|mlflow"
                '''
            }
        }
        
        stage('Data Validation') {
            steps {
                echo '📊 Running data validation tests...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f tests/test_data.py ]; then
                        python tests/test_data.py
                    else
                        echo "⚠️  Data tests not found, checking data files..."
                        ls -lh data/processed/ || echo "No processed data directory"
                        ls -lh data/raw/ || echo "No raw data directory"
                    fi
                '''
            }
        }
        stage('Generate Missing Data') {
	    steps {
		echo '📊 Checking and generating missing data files...'
		sh '''
		    . venv/bin/activate
		    if [ ! -f data/raw/wearable_health.csv ]; then
		        echo "Generating wearable_health.csv..."
		        python src/data_ingestion/collect_data.py
		    else
		        echo "Data files already exist"
		    fi
		'''
	    }
	}
        stage('Train Models') {
            steps {
                echo '🤖 Training machine learning models...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f src/models/baseline_models.py ]; then
                        echo "Training baseline models..."
                        python src/models/baseline_models.py
                        
                        echo "\n📊 Model outputs:"
                        ls -lh models/ || echo "Models directory not created yet"
                    else
                        echo "⚠️  baseline_models.py not found"
                        exit 1
                    fi
                '''
            }
        }
        
        stage('Model Validation') {
            steps {
                echo '✅ Validating trained models...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f tests/test_model.py ]; then
                        python tests/test_model.py
                    else
                        echo "⚠️  Model tests not found"
                        echo "Checking if model files exist..."
                        find models -name "*.pkl" -o -name "*.joblib" || echo "No model files found"
                    fi
                '''
            }
        }
        
        stage('Data Drift Monitoring') {
            steps {
                echo '🔍 Running data drift detection...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f src/mlops/drift_monitor.py ]; then
                        echo "Running drift monitor..."
                        python src/mlops/drift_monitor.py
                        
                        echo "\n📈 Drift detection outputs:"
                        if [ -f models/drift_summary.json ]; then
                            echo "✅ Drift summary generated"
                            cat models/drift_summary.json
                        else
                            echo "⚠️  drift_summary.json not found"
                        fi
                        
                        if [ -f docs/drift_report.html ]; then
                            echo "✅ HTML drift report generated"
                            ls -lh docs/drift_report.html
                        else
                            echo "⚠️  drift_report.html not found"
                        fi
                    else
                        echo "⚠️  drift_monitor.py not found at src/mlops/drift_monitor.py"
                        exit 1
                    fi
                '''
            }
        }
        
        stage('Analyze Drift Results') {
            steps {
                echo '📊 Analyzing drift detection results...'
                script {
                    def driftSummaryPath = "${WORKSPACE}/models/drift_summary.json"
                    
                    if (fileExists(driftSummaryPath)) {
                        def driftSummary = readJSON file: driftSummaryPath
                        
                        echo "═══════════════════════════════════════"
                        echo "   DRIFT DETECTION SUMMARY"
                        echo "═══════════════════════════════════════"
                        echo "Timestamp: ${driftSummary.timestamp}"
                        echo "Reference Size: ${driftSummary.reference_size}"
                        echo "Current Size: ${driftSummary.current_size}"
                        echo "Features Monitored: ${driftSummary.features_monitored}"
                        
                        if (driftSummary.dataset_drift == true) {
                            echo "⚠️  DRIFT DETECTED: YES"
                            echo "Drift Share: ${driftSummary.drift_share * 100}%"
                            echo "Drifted Features: ${driftSummary.drifted_features.size()}"
                            
                            if (driftSummary.drifted_features.size() > 0) {
                                echo "\n🚨 Drifted Features:"
                                driftSummary.drifted_features.each { feature ->
                                    echo "  - ${feature.feature}: ${feature.drift_score}"
                                }
                            }
                            
                            echo "\n⚠️  RECOMMENDATION: Model retraining is recommended"
                            currentBuild.result = 'UNSTABLE'
                        } else {
                            echo "✅ DRIFT DETECTED: NO"
                            echo "✅ Model performance should remain stable"
                        }
                        echo "═══════════════════════════════════════"
                    } else {
                        echo "❌ Drift summary file not found"
                        currentBuild.result = 'UNSTABLE'
                    }
                }
            }
        }
        
        stage('Run MLflow Tracking') {
            steps {
                echo '📈 Logging experiments to MLflow...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f src/mlops/mlflow_tracking.py ]; then
                        python src/mlops/mlflow_tracking.py
                        
                        echo "\n📊 MLflow runs:"
                        if [ -d mlruns ]; then
                            ls -la mlruns/
                            find mlruns -type f -name "*.json" | head -5
                        else
                            echo "⚠️  mlruns directory not found"
                        fi
                    else
                        echo "⚠️  mlflow_tracking.py not found"
                    fi
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                script {
                    if (fileExists('Dockerfile')) {
                        sh """
                            docker build -t ${DOCKER_IMAGE}:${BUILD_NUMBER} .
                            docker tag ${DOCKER_IMAGE}:${BUILD_NUMBER} ${DOCKER_IMAGE}:latest
                            
                            echo "\n🐳 Docker images:"
                            docker images | grep ${DOCKER_IMAGE}
                        """
                    } else {
                        echo "⚠️  Dockerfile not found, skipping Docker build"
                    }
                }
            }
        }
        
        stage('Stop Old Containers') {
            steps {
                echo '🛑 Stopping old API containers...'
                script {
                    sh """
                        docker stop ${DOCKER_IMAGE} 2>/dev/null || echo "No container to stop"
                        docker rm ${DOCKER_IMAGE} 2>/dev/null || echo "No container to remove"
                        echo "✅ Old containers cleaned"
                    """
                }
            }
        }
        
        stage('Deploy API') {
            steps {
                echo '🚀 Deploying API container...'
                script {
                    sh """
                        docker run -d \
                            --name ${DOCKER_IMAGE} \
                            -p ${API_PORT}:${API_PORT} \
                            -v ${WORKSPACE}/models:/app/models:ro \
                            -v ${WORKSPACE}/data:/app/data:ro \
                            ${DOCKER_IMAGE}:latest
                        
                        echo "\n📦 Container status:"
                        docker ps | grep ${DOCKER_IMAGE}
                        
                        echo "\n⏳ Waiting for API to start..."
                        sleep 5
                    """
                }
            }
        }
        
        stage('API Health Check') {
            steps {
                echo '🏥 Checking API health...'
                sh '''
                    max_attempts=10
                    attempt=0
                    
                    while [ $attempt -lt $max_attempts ]; do
                        if curl -s http://localhost:${API_PORT}/health > /dev/null; then
                            echo "✅ API is healthy"
                            curl -s http://localhost:${API_PORT}/health | python3 -m json.tool
                            exit 0
                        fi
                        attempt=$((attempt + 1))
                        echo "Attempt $attempt/$max_attempts - waiting..."
                        sleep 3
                    done
                    
                    echo "❌ API health check failed"
                    docker logs ${DOCKER_IMAGE}
                    exit 1
                '''
            }
        }
        
        stage('API Integration Tests') {
            steps {
                echo '🧪 Running API integration tests...'
                sh '''
                    . venv/bin/activate
                    
                    if [ -f tests/test_api.py ]; then
                        python tests/test_api.py
                    else
                        echo "⚠️  API tests not found, running basic test..."
                        python3 -c "
import requests
import json

url = 'http://localhost:${API_PORT}/predict'
data = {
    'pm25': 50, 'pm10': 80, 'aqi': 150,
    'temperature': 25, 'humidity': 60, 'precipitation': 2,
    'heart_rate_avg': 75, 'sleep_hours': 7,
    'spo2': 98, 'stress_level': 5
}

response = requests.post(url, json=data)
print(f'Status: {response.status_code}')
print(f'Response: {json.dumps(response.json(), indent=2)}')

if response.status_code == 200:
    print('✅ API test passed')
else:
    raise Exception('API test failed')
"
                    fi
                '''
            }
        }
        
        stage('Performance Benchmark') {
            steps {
                echo '⚡ Running performance benchmarks...'
                sh '''
                    . venv/bin/activate
                    python3 -c "
import requests
import time
import statistics

url = 'http://localhost:${API_PORT}/predict'
data = {
    'pm25': 50, 'pm10': 80, 'aqi': 150,
    'temperature': 25, 'humidity': 60, 'precipitation': 2,
    'heart_rate_avg': 75, 'sleep_hours': 7,
    'spo2': 98, 'stress_level': 5
}

print('⚡ Running 100 requests...')
times = []
success_count = 0

for i in range(100):
    try:
        start = time.time()
        r = requests.post(url, json=data, timeout=5)
        elapsed = time.time() - start
        times.append(elapsed)
        if r.status_code == 200:
            success_count += 1
    except Exception as e:
        print(f'Request {i+1} failed: {e}')

if success_count > 0:
    print(f'\\n📊 Performance Results:')
    print(f'Success Rate: {success_count}/100 ({success_count}%)')
    print(f'Avg Response: {statistics.mean(times)*1000:.0f}ms')
    print(f'Min: {min(times)*1000:.0f}ms')
    print(f'Max: {max(times)*1000:.0f}ms')
    if len(times) >= 20:
        print(f'P95: {statistics.quantiles(times, n=20)[18]*1000:.0f}ms')
    print('✅ Performance benchmark passed')
else:
    raise Exception('Performance benchmark failed - no successful requests')
"
                '''
            }
        }
        
        stage('Generate Report') {
            steps {
                echo '📄 Generating comprehensive build report...'
                sh '''
                    . venv/bin/activate
                    
                    cat > build_report.txt << 'EOF'
═══════════════════════════════════════════════════════════
                    BUILD REPORT
═══════════════════════════════════════════════════════════
EOF
                    
                    echo "Build Number: ${BUILD_NUMBER}" >> build_report.txt
                    echo "Build Date: $(date)" >> build_report.txt
                    echo "Jenkins URL: ${BUILD_URL}" >> build_report.txt
                    git log -1 --oneline >> build_report.txt 2>/dev/null || echo "Not a git repo" >> build_report.txt
                    echo "" >> build_report.txt
                    
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    echo "                MODEL PERFORMANCE" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    
                    if [ -f models/baseline_results.json ]; then
                        python3 -c "
import json
with open('models/baseline_results.json') as f:
    results = json.load(f)
    for model, metrics in results.items():
        print(f'{model}:')
        print(f'  Accuracy: {metrics.get(\"accuracy\", 0):.4f}')
        print(f'  ROC-AUC: {metrics.get(\"roc_auc\", 0):.4f}')
        print()
" >> build_report.txt
                    else
                        echo "⚠️  Model results not found" >> build_report.txt
                    fi
                    
                    echo "" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    echo "                DRIFT DETECTION" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    
                    if [ -f models/drift_summary.json ]; then
                        cat models/drift_summary.json >> build_report.txt
                    else
                        echo "⚠️  Drift summary not found" >> build_report.txt
                    fi
                    
                    echo "" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    echo "                DOCKER IMAGE" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    docker images | grep ${DOCKER_IMAGE} >> build_report.txt 2>/dev/null || echo "No images" >> build_report.txt
                    
                    echo "" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    echo "                API STATUS" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    curl -s http://localhost:${API_PORT}/health >> build_report.txt 2>/dev/null || echo "API not reachable" >> build_report.txt
                    
                    echo "" >> build_report.txt
                    echo "═══════════════════════════════════════════════════════════" >> build_report.txt
                    
                    cat build_report.txt
                '''
                
                archiveArtifacts artifacts: 'build_report.txt, models/drift_summary.json, docs/drift_report.html', 
                                 allowEmptyArchive: true,
                                 fingerprint: true
            }
        }
    }
    
    post {
        success {
            echo '''
            ✅ ═══════════════════════════════════════════════════════════
            ✅              PIPELINE COMPLETED SUCCESSFULLY
            ✅ ═══════════════════════════════════════════════════════════
            '''
            sh '''
                echo "✅ All tests passed"
                echo "✅ Models trained and validated"
                echo "✅ Drift monitoring completed"
                echo "✅ Docker image built"
                echo "✅ API deployed and tested"
                echo ""
                echo "🌐 API URLs:"
                echo "   - Health: http://localhost:${API_PORT}/health"
                echo "   - Docs: http://localhost:${API_PORT}/docs"
                echo "   - Predict: http://localhost:${API_PORT}/predict"
                echo ""
                echo "📊 Artifacts:"
                echo "   - Build Report: ${BUILD_URL}artifact/build_report.txt"
                echo "   - Drift Report: ${BUILD_URL}artifact/docs/drift_report.html"
            '''
        }
        
        unstable {
            echo '''
            ⚠️  ═══════════════════════════════════════════════════════════
            ⚠️         PIPELINE COMPLETED WITH WARNINGS
            ⚠️  ═══════════════════════════════════════════════════════════
            '''
            sh '''
                echo "⚠️  Data drift detected or tests unstable"
                echo "⚠️  Review drift report and consider model retraining"
                echo ""
                if [ -f models/drift_summary.json ]; then
                    echo "📊 Drift Summary:"
                    cat models/drift_summary.json
                fi
            '''
        }
        
        failure {
            echo '''
            ❌ ═══════════════════════════════════════════════════════════
            ❌                  PIPELINE FAILED
            ❌ ═══════════════════════════════════════════════════════════
            '''
            sh '''
                echo "❌ Pipeline failed at stage: ${STAGE_NAME}"
                echo "❌ Build Number: ${BUILD_NUMBER}"
                echo "❌ Check logs for details: ${BUILD_URL}console"
                echo ""
                echo "🧹 Cleaning up..."
                
                # Stop and remove failed containers
                docker stop ${DOCKER_IMAGE} 2>/dev/null || true
                docker rm ${DOCKER_IMAGE} 2>/dev/null || true
                
                # Show Docker logs if container exists
                docker logs ${DOCKER_IMAGE} 2>/dev/null || echo "No container logs available"
            '''
        }
        
        always {
            echo '🧹 Post-build cleanup...'
            sh '''
                # Archive important files
                mkdir -p archive
                cp models/*.json archive/ 2>/dev/null || true
                cp docs/*.html archive/ 2>/dev/null || true
                
                # Clean temporary files
                rm -rf venv/.pytest_cache 2>/dev/null || true
                rm -rf .pytest_cache 2>/dev/null || true
                
                echo "✅ Cleanup complete"
            '''
            
            // Clean workspace after archiving
            cleanWs(
                deleteDirs: true,
                disableDeferredWipeout: true,
                notFailBuild: true,
                patterns: [
                    [pattern: 'venv/**', type: 'INCLUDE'],
                    [pattern: '.pytest_cache/**', type: 'INCLUDE']
                ]
            )
        }
    }
}
