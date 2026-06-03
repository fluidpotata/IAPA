pipeline {
    agent any

    environment {
        IMAGE_NAME = "bracu-rag"
        IMAGE_TAG = "${BUILD_NUMBER}"
        REGISTRY = "potata"
        GEMINI_API_KEY = credentials("gemini-api-key")
    }

    stages {
        stage("Checkout") {
            steps {
                checkout scm
            }
        }

        stage("Test"){
            steps {
                sh """
                    python -m pytest src/tests/ -v --tb=short
                """
            }
        }

        stage("Build"){
            steps {
                sh """
                    docker build \
                        -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
                        -t ${REGISTRY}/${IMAGE_NAME}:latest \
                        .
                """
            }
        }

        stage("Push"){
            withCredentials([usernamePassword(
                    credentialsId: "dockerhub-creds",
                    usernameVariable: "DOCKER_USER",
                    passwordVariable: "DOCKER_PASS"
                )]) {
                    sh """
                        echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
                        docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${REGISTRY}/${IMAGE_NAME}:latest
                    """
                }
        }

        stage("Re-ingest"){
            when {
                changeset "docs/**/*.md"
            }
            steps{
                sh """
                    docker run --rm \
                        -e GEMINI_API_KEY=${GEMINI_API_KEY} \
                        -v bracu-rag-chroma:/app/chroma_db \
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
                        python src/ingest.py
                """
            }
        }

        stage("Deploy"){
            steps{
                sh """
                    kubectl set image deployment/bracu-rag \
                        bracu-rag=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

                    kubectl rollout status deployment deployment/bracu-rag \
                        --timeout=120s
                """
            }
        }

        stage("Verify") {
            steps {
                sh """
                    sleep 10

                    SERVICE_IP=\$(kubectl get service bracu-rag-service \
                        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

                    curl --fail http://\$SERVICE_IP/health
                """
            }
        }
    }

    post {
        success {
            echo "Deployment successful. Build ${BUILD_NUMBER} is live."
        }
        failure {
            sh """
                kubectl rollout undo deployment/bracu-rag
            """
            echo "Deployment failed. Rolled back to previous version."
        }
        always {
            sh "docker rmi ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} || true"
        }
    }
}