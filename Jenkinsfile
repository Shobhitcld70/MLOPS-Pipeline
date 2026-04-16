pipeline {
    agent any

    environment {
        IMAGE_NAME     = "mlops-pipeline"
        IMAGE_TAG      = "${BUILD_NUMBER}"
        REGISTRY       = "your-dockerhub-username"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.GIT_BRANCH} | Commit: ${env.GIT_COMMIT[0..6]}"
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest tests/ -v --tb=short --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
                failure {
                    error "Tests failed — aborting pipeline."
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ."
                sh "docker tag ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:latest"
            }
        }

        stage('Push to Registry') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
                        docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${REGISTRY}/${IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    sed -i 's|IMAGE_PLACEHOLDER|${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g' \
                        configs/k8s-deployment.yaml
                    kubectl apply -f configs/k8s-deployment.yaml
                    kubectl rollout status deployment/mlops-api --timeout=120s
                """
            }
        }

        stage('Trigger ML Pipeline') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    curl -s -X POST http://localhost:5000/pipeline/run \
                         -H "Content-Type: application/json" \
                         -d '{"retrain": false}' | python3 -m json.tool
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded — build ${BUILD_NUMBER} deployed."
        }
        failure {
            echo "Pipeline FAILED at stage: ${env.STAGE_NAME}"
        }
        always {
            sh "docker rmi ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} || true"
            cleanWs()
        }
    }
}
