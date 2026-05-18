
import argparse
import time
import signal
import sys

import ray
from ray import serve

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from serve import app as fastapi_app



@serve.deployment(
    num_replicas          = 1,       
    ray_actor_options     = {
        "num_cpus": 1,
        "num_gpus": 0,
    },
    max_ongoing_requests  = 10,      
    health_check_period_s  = 10,     
    health_check_timeout_s = 30,     
)
@serve.ingress(fastapi_app)
class DemandForecastingDeployment:
    def __init__(self):
        import os
        from serve import load_all_models
        replica_id = os.environ.get("RAY_SERVE_REPLICA_ID", "unknown")
        print(f"[Ray Serve] Replica started: {replica_id}")
        load_all_models()

    def check_health(self):
        from serve import MODELS
        if not MODELS:
            raise RuntimeError("No models loaded — replica unhealthy")


# ── Build application ────────────────────────────────────────────────────────

def build_app(num_replicas: int = 1) -> serve.Application:
    deployment = DemandForecastingDeployment.options(
        num_replicas=num_replicas
    )
    return deployment.bind()


# ── Run ──────────────────────────────────────────────────────────────────────

def run(num_replicas: int = 1):
    print(f"Initializing Ray...")
    ray.init(ignore_reinit_error=True)

    print(f"Starting Ray Serve with {num_replicas} replica(s)...")
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
        }
    )

    # Deploy application
    app = build_app(num_replicas=num_replicas)
    serve.run(app, route_prefix="/")

    print("\n" + "=" * 55)
    print("  Ray Serve is running!")
    print("=" * 55)
    print(f"  API          : http://localhost:8000")
    print(f"  Swagger UI   : http://localhost:8000/docs")
    print(f"  Replicas     : {num_replicas}")
    print(f"  Ray Dashboard: http://localhost:8265")
    print("=" * 55)
    print("\nPress Ctrl+C to stop...\n")

    def shutdown_handler(sig, frame):
        print("\nShutting down Ray Serve...")
        serve.shutdown()
        ray.shutdown()
        print("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        try:
            status = serve.status()
            apps   = status.applications

            for app_name, app_status in apps.items():
                app_state = app_status.status
                deployments = app_status.deployments

                for dep_name, dep_status in deployments.items():
                    replicas = dep_status.replica_states
                    running  = replicas.get("RUNNING", 0)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{dep_name} — {app_state} | "
                        f"replicas running: {running}"
                    )

            time.sleep(30)

        except Exception as e:
            print(f"Status check error: {e}")
            time.sleep(30)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI app with Ray Serve")
    parser.add_argument(
        "--replicas", type=int, default=1,
        help="Number of replicas (default: 1). Increase if needed to handle multiple concurrent requests."
    )
    args = parser.parse_args()

    run(num_replicas=args.replicas)