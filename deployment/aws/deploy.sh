#!/bin/bash

# Bridgestone Vehicle Safety System - AWS Deployment Script
# Automates deployment to AWS using CloudFormation and supporting services

set -e  # Exit on any error

# Configuration
STACK_NAME="bridgestone-vehicle-safety"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGION="${AWS_REGION:-us-east-1}"
PROFILE="${AWS_PROFILE:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged into AWS
    if ! aws sts get-caller-identity --profile $PROFILE &> /dev/null; then
        log_error "Not logged into AWS or invalid profile. Please run 'aws configure' first."
        exit 1
    fi
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        log_warning "jq is not installed. Some features may not work properly."
    fi
    
    log_success "Prerequisites check passed"
}

# Create S3 bucket for deployment artifacts
create_deployment_bucket() {
    BUCKET_NAME="bridgestone-vehicle-safety-deployment-${ENVIRONMENT}"
    
    log_info "Creating deployment bucket: $BUCKET_NAME"
    
    # Check if bucket exists
    if aws s3 ls "s3://$BUCKET_NAME" --profile $PROFILE 2>&1 | grep -q 'NoSuchBucket'; then
        # Create bucket
        if [ "$REGION" = "us-east-1" ]; then
            aws s3 mb "s3://$BUCKET_NAME" --profile $PROFILE
        else
            aws s3 mb "s3://$BUCKET_NAME" --region $REGION --profile $PROFILE
        fi
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket $BUCKET_NAME \
            --versioning-configuration Status=Enabled \
            --profile $PROFILE
        
        log_success "Deployment bucket created: $BUCKET_NAME"
    else
        log_info "Deployment bucket already exists: $BUCKET_NAME"
    fi
    
    echo $BUCKET_NAME
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile $PROFILE)
    
    # ECR repository name
    REPO_NAME="bridgestone/vehicle-safety"
    ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION --profile $PROFILE 2>/dev/null || {
        log_info "Creating ECR repository..."
        aws ecr create-repository --repository-name $REPO_NAME --region $REGION --profile $PROFILE
    }
    
    # Login to ECR
    aws ecr get-login-password --region $REGION --profile $PROFILE | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    log_info "Building Docker image..."
    cd "$(dirname "$0")/../.."
    docker build -t $REPO_NAME:latest -f deployment/Dockerfile .
    
    # Tag and push image
    docker tag $REPO_NAME:latest $ECR_URI:latest
    docker tag $REPO_NAME:latest $ECR_URI:$(date +%Y%m%d-%H%M%S)
    
    log_info "Pushing Docker image to ECR..."
    docker push $ECR_URI:latest
    docker push $ECR_URI:$(date +%Y%m%d-%H%M%S)
    
    log_success "Docker image pushed: $ECR_URI:latest"
    echo $ECR_URI
}

# Package Lambda function
package_lambda() {
    log_info "Packaging Lambda function..."
    
    cd "$(dirname "$0")"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    
    # Copy Lambda function
    cp lambda_function.py $TEMP_DIR/
    
    # Install dependencies
    pip install -r ../../requirements.txt -t $TEMP_DIR/ --quiet
    
    # Create deployment package
    cd $TEMP_DIR
    zip -r ../lambda_deployment.zip . > /dev/null
    
    LAMBDA_PACKAGE="$(dirname "$TEMP_DIR")/lambda_deployment.zip"
    
    # Upload to S3
    BUCKET_NAME=$(create_deployment_bucket)
    aws s3 cp $LAMBDA_PACKAGE s3://$BUCKET_NAME/lambda/lambda_deployment.zip --profile $PROFILE
    
    # Cleanup
    rm -rf $TEMP_DIR
    rm $LAMBDA_PACKAGE
    
    log_success "Lambda function packaged and uploaded"
    echo "s3://$BUCKET_NAME/lambda/lambda_deployment.zip"
}

# Upload CloudFormation templates
upload_templates() {
    log_info "Uploading CloudFormation templates..."
    
    BUCKET_NAME=$(create_deployment_bucket)
    
    # Upload main template
    aws s3 cp cloudformation.yaml s3://$BUCKET_NAME/templates/cloudformation.yaml --profile $PROFILE
    
    log_success "CloudFormation templates uploaded"
    echo "s3://$BUCKET_NAME/templates/cloudformation.yaml"
}

# Deploy CloudFormation stack
deploy_stack() {
    log_info "Deploying CloudFormation stack..."
    
    TEMPLATE_URL=$(upload_templates)
    
    # Stack parameters
    PARAMETERS="ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --profile $PROFILE 2>/dev/null; then
        log_info "Updating existing stack..."
        aws cloudformation update-stack \
            --stack-name $STACK_NAME \
            --template-url $TEMPLATE_URL \
            --parameters $PARAMETERS \
            --capabilities CAPABILITY_IAM \
            --region $REGION \
            --profile $PROFILE
        
        log_info "Waiting for stack update to complete..."
        aws cloudformation wait stack-update-complete \
            --stack-name $STACK_NAME \
            --region $REGION \
            --profile $PROFILE
    else
        log_info "Creating new stack..."
        aws cloudformation create-stack \
            --stack-name $STACK_NAME \
            --template-url $TEMPLATE_URL \
            --parameters $PARAMETERS \
            --capabilities CAPABILITY_IAM \
            --region $REGION \
            --profile $PROFILE
        
        log_info "Waiting for stack creation to complete..."
        aws cloudformation wait stack-create-complete \
            --stack-name $STACK_NAME \
            --region $REGION \
            --profile $PROFILE
    fi
    
    log_success "CloudFormation stack deployed successfully"
}

# Update Lambda function code
update_lambda() {
    log_info "Updating Lambda function code..."
    
    LAMBDA_PACKAGE=$(package_lambda)
    
    # Get function name from stack outputs
    FUNCTION_NAME=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --profile $PROFILE \
        --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionName`].OutputValue' \
        --output text 2>/dev/null || echo "vehicle-safety-inference-$ENVIRONMENT")
    
    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --s3-bucket $(echo $LAMBDA_PACKAGE | cut -d'/' -f3) \
        --s3-key lambda/lambda_deployment.zip \
        --region $REGION \
        --profile $PROFILE
    
    log_success "Lambda function updated"
}

# Deploy models to S3
deploy_models() {
    log_info "Deploying models to S3..."
    
    MODELS_BUCKET="bridgestone-vehicle-safety-$ENVIRONMENT"
    
    # Create models bucket if it doesn't exist
    if aws s3 ls "s3://$MODELS_BUCKET" --profile $PROFILE 2>&1 | grep -q 'NoSuchBucket'; then
        if [ "$REGION" = "us-east-1" ]; then
            aws s3 mb "s3://$MODELS_BUCKET" --profile $PROFILE
        else
            aws s3 mb "s3://$MODELS_BUCKET" --region $REGION --profile $PROFILE
        fi
    fi
    
    # Upload models (if they exist)
    cd "$(dirname "$0")/../.."
    
    if [ -d "data/models" ]; then
        aws s3 sync data/models/ s3://$MODELS_BUCKET/models/ --profile $PROFILE
        log_success "Models deployed to S3"
    else
        log_warning "No models directory found. Skipping model deployment."
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get API Gateway URL
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --profile $PROFILE \
        --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ ! -z "$API_URL" ]; then
        # Test API Gateway health
        log_info "Testing API Gateway health..."
        if curl -s -f "$API_URL/health" > /dev/null; then
            log_success "API Gateway health check passed"
        else
            log_warning "API Gateway health check failed"
        fi
    fi
    
    # Get Load Balancer URL
    ALB_URL=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --profile $PROFILE \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerUrl`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ ! -z "$ALB_URL" ]; then
        log_info "Testing Load Balancer health..."
        if curl -s -f "$ALB_URL/health" > /dev/null; then
            log_success "Load Balancer health check passed"
        else
            log_warning "Load Balancer health check failed"
        fi
    fi
}

# Print deployment summary
print_summary() {
    log_info "Deployment Summary"
    echo "===================="
    
    # Get stack outputs
    OUTPUTS=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --profile $PROFILE \
        --query 'Stacks[0].Outputs' 2>/dev/null)
    
    if [ ! -z "$OUTPUTS" ] && [ "$OUTPUTS" != "null" ]; then
        echo "$OUTPUTS" | jq -r '.[] | "\(.OutputKey): \(.OutputValue)"' 2>/dev/null || {
            echo "Stack outputs available in AWS Console"
        }
    else
        log_warning "No stack outputs available"
    fi
    
    echo "===================="
    log_success "Deployment completed successfully!"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Help function
show_help() {
    echo "Bridgestone Vehicle Safety AWS Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy          Full deployment (default)"
    echo "  update-lambda   Update Lambda function only"
    echo "  update-models   Update models only"
    echo "  health-check    Run health checks only"
    echo "  cleanup         Clean up resources"
    echo ""
    echo "Options:"
    echo "  -e, --environment   Environment (development, staging, production)"
    echo "  -r, --region        AWS region (default: us-east-1)"
    echo "  -p, --profile       AWS profile (default: default)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT         Deployment environment"
    echo "  AWS_REGION         AWS region"
    echo "  AWS_PROFILE        AWS profile"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  $0 -e staging deploy"
    echo "  $0 update-lambda"
    echo "  $0 health-check"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy|update-lambda|update-models|health-check|cleanup)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default command
COMMAND="${COMMAND:-deploy}"

# Main execution
main() {
    log_info "Starting Bridgestone Vehicle Safety deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $REGION"
    log_info "Profile: $PROFILE"
    log_info "Command: $COMMAND"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    case $COMMAND in
        deploy)
            check_prerequisites
            deploy_stack
            update_lambda
            deploy_models
            run_health_checks
            print_summary
            ;;
        update-lambda)
            check_prerequisites
            update_lambda
            ;;
        update-models)
            check_prerequisites
            deploy_models
            ;;
        health-check)
            run_health_checks
            ;;
        cleanup)
            log_info "Cleanup functionality not implemented yet"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
