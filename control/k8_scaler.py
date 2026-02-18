from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

def load_config():
    try:
        config.load_kube_config()
        print("Loaded local kubeconfig")
    except ConfigException:
        config.load_incluster_config()
        print("Loaded in-cluster config")

def get_apps_api():
    return client.AppsV1Api()

def get_current_replicas(deployment_name, namespace="default"):
    api = get_apps_api()

    try:
        deployment = api.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        replicas = deployment.spec.replicas
        print(f"Current desired replicas: {replicas}")
        return replicas

    except client.exceptions.ApiException as e:
        print(f"API Error while fetching replicas: {e}")
        return None

def set_replicas(deployment_name, replicas, namespace="default"):
    api = get_apps_api()

    print(f"Requested replica count: {replicas}")

    try:
        body = {
            "spec": {
                "replicas": replicas
            }
        }

        api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )

        # Read back to verify
        updated = api.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        observed = updated.status.ready_replicas
        print(f"Observed replica count after update: {observed}")

        return observed

    except client.exceptions.ApiException as e:
        print(f"API Error while scaling: {e}")
        return None

if __name__ == "__main__":
    load_config()

    deployment = "my-app"

    print("Before scaling:")
    get_current_replicas(deployment)

    print("\nScaling to 3 replicas...")
    set_replicas(deployment, 3)

    print("\nAfter scaling:")
    get_current_replicas(deployment)
