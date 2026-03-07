from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
from config.logger import setup_logger

logger = setup_logger("autoscale.k8_scaler")

apps_api = None


def load_config():
    global apps_api

    try:
        config.load_kube_config()
        logger.info("Loaded local kubeconfig")
    except ConfigException:
        config.load_incluster_config()
        logger.info("Loaded in-cluster config")

    apps_api = client.AppsV1Api()


def get_current_replicas(deployment_name, namespace="default"):

    try:
        deployment = apps_api.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        replicas = deployment.spec.replicas
        logger.info(f"Current desired replicas: {replicas}")
        return replicas

    except client.exceptions.ApiException as e:
        logger.error(f"API Error while fetching replicas: {e}")
        return None


def set_replicas(deployment_name, replicas, namespace="default"):

    logger.info(f"Requested replica count: {replicas}")

    try:
        body = {
            "spec": {
                "replicas": replicas
            }
        }

        apps_api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )

        logger.info(f"Scale request accepted: {replicas}")
        return replicas

    except client.exceptions.ApiException as e:
        logger.error(f"API Error while scaling: {e}")
        return None


if __name__ == "__main__":

    load_config()

    deployment = "my-app"

    logger.info("Before scaling:")
    get_current_replicas(deployment)

    logger.info("\nScaling to 3 replicas...")
    set_replicas(deployment, 3)

    logger.info("\nAfter scaling:")
    get_current_replicas(deployment)