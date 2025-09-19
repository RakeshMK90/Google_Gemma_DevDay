#!/usr/bin/env python3
"""
Podman Container Manager for Gemma3 Model Serving
Replaces Ollama with containerized model serving using ramalama
"""

import subprocess
import time
import requests
import json
import os
import signal
from typing import Optional, Dict, List
from pathlib import Path

class PodmanManager:
    def __init__(self, 
                 container_name: str = "ramalama_gemma3",
                 port: int = 8080,
                 model_path: str = "/mnt/models/gemma-3-4b-it-Q4_K_M.gguf",
                 mmproj_path: str = "/mnt/models/mmproj-model-f16.gguf"):
        """
        Initialize Podman Manager for Gemma3 model serving
        
        Args:
            container_name: Name for the container
            port: Port to expose the model server
            model_path: Path to the model file inside container
            mmproj_path: Path to the multimodal projection model
        """
        self.container_name = container_name
        self.port = port
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.api_url = f"http://127.0.0.1:{port}"
        
        # Default configuration based on your working podman command
        self.default_config = {
            "image": "ghcr.io/rakeshmk90/ramalama-jetson:latest",
            "labels": {
                "ai.ramalama.model": "hf://ggml-org/gemma-3-4b-it-GGUF",
                "ai.ramalama.engine": "podman",
                "ai.ramalama.runtime": "llama.cpp",
                "ai.ramalama.port": str(port),
                "ai.ramalama.command": "run"
            },
            "devices": ["/dev/dri", "nvidia.com/gpu=all"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HOME": "/tmp"
            },
            "security_opts": ["label=disable", "no-new-privileges"],
            "mounts": [
                {
                    "type": "bind",
                    "src": "/home/rakesh/.local/share/ramalama/store/huggingface/ggml-org/gemma-3-4b-it-GGUF/blobs/sha256-882e8d2db44dc554fb0ea5077cb7e4bc49e7342a1f0da57901c0802ea21a0863",
                    "dest": "/mnt/models/gemma-3-4b-it-Q4_K_M.gguf",
                    "readonly": True
                },
                {
                    "type": "bind", 
                    "src": "/home/rakesh/.local/share/ramalama/store/huggingface/ggml-org/gemma-3-4b-it-GGUF/blobs/sha256-8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb",
                    "dest": "/mnt/models/mmproj-model-f16.gguf",
                    "readonly": True
                }
            ],
            "server_args": [
                "--port", str(port),
                "--model", model_path,
                "--no-warmup",
                "--mmproj", mmproj_path,
                "--log-colors",
                "--alias", "ggml-org/gemma-3-4b-it-GGUF",
                "--ctx-size", "2048",
                "--temp", "0.8",
                "--cache-reuse", "256",
                "-v",
                "--flash-attn",
                "-ngl", "999",
                "--threads", "3",
                "--host", "0.0.0.0"
            ]
        }
    
    def _run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return result"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                check=False
            )
            return result
        except Exception as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e}")
            raise
    
    def is_container_running(self) -> bool:
        """Check if the container is currently running"""
        try:
            result = self._run_command(["podman", "ps", "--format", "json"])
            if result.returncode == 0:
                containers = json.loads(result.stdout) if result.stdout.strip() else []
                for container in containers:
                    if container.get("Names", [{}])[0] == self.container_name:
                        return True
            return False
        except Exception:
            return False
    
    def is_container_exists(self) -> bool:
        """Check if the container exists (running or stopped)"""
        try:
            result = self._run_command(["podman", "ps", "-a", "--format", "json"])
            if result.returncode == 0:
                containers = json.loads(result.stdout) if result.stdout.strip() else []
                for container in containers:
                    if container.get("Names", [{}])[0] == self.container_name:
                        return True
            return False
        except Exception:
            return False
    
    def stop_container(self) -> bool:
        """Stop the container if running"""
        try:
            if not self.is_container_running():
                print(f"Container {self.container_name} is not running")
                return True
            
            print(f"Stopping container {self.container_name}...")
            result = self._run_command(["podman", "stop", self.container_name])
            
            if result.returncode == 0:
                print(f"Container {self.container_name} stopped successfully")
                return True
            else:
                print(f"Failed to stop container: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error stopping container: {e}")
            return False
    
    def remove_container(self) -> bool:
        """Remove the container"""
        try:
            if not self.is_container_exists():
                print(f"Container {self.container_name} does not exist")
                return True
            
            # Stop first if running
            self.stop_container()
            
            print(f"Removing container {self.container_name}...")
            result = self._run_command(["podman", "rm", self.container_name])
            
            if result.returncode == 0:
                print(f"Container {self.container_name} removed successfully")
                return True
            else:
                print(f"Failed to remove container: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error removing container: {e}")
            return False
    
    def start_container(self, detached: bool = True) -> bool:
        """Start the container with the model server"""
        try:
            # Stop and remove existing container if it exists
            if self.is_container_exists():
                print("Existing container found, removing...")
                self.remove_container()
            
            # Build podman run command
            cmd = ["podman", "run"]
            
            # Add labels
            for key, value in self.default_config["labels"].items():
                cmd.extend(["--label", f"{key}={value}"])
            
            # Add devices
            for device in self.default_config["devices"]:
                cmd.extend(["--device", device])
            
            # Add environment variables
            for key, value in self.default_config["environment"].items():
                cmd.extend(["-e", f"{key}={value}"])
            
            # Add security options
            for opt in self.default_config["security_opts"]:
                cmd.extend(["--security-opt", opt])
            
            # Add port mapping
            cmd.extend(["-p", f"{self.port}:{self.port}"])
            
            # Add other options
            cmd.extend(["--group-add", "keep-groups"])
            cmd.extend(["--pull", "newer"])
            cmd.extend(["-t"])
            if detached:
                cmd.extend(["-d"])
            cmd.extend(["-i"])
            cmd.extend(["--label", "ai.ramalama"])
            cmd.extend(["--name", self.container_name])
            cmd.extend(["--init"])
            
            # Add mounts
            for mount in self.default_config["mounts"]:
                mount_str = f"type={mount['type']},src={mount['src']},destination={mount['dest']}"
                if mount.get('readonly'):
                    mount_str += ",ro"
                cmd.extend(["--mount", mount_str])
            
            # Add image
            cmd.append(self.default_config["image"])
            
            # Add server command and arguments
            cmd.append("llama-server")
            cmd.extend(self.default_config["server_args"])
            
            print(f"Starting container {self.container_name}...")
            print(f"Command: {' '.join(cmd)}")
            
            result = self._run_command(cmd)
            
            if result.returncode == 0:
                print(f"Container {self.container_name} started successfully")
                if detached:
                    # Wait a moment for container to initialize
                    time.sleep(2)
                    return self.wait_for_server()
                return True
            else:
                print(f"Failed to start container: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error starting container: {e}")
            return False
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """Wait for the model server to be ready"""
        print("Waiting for model server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print("\n✓ Model server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                print(f"\n[+{elapsed}s] Still loading model...")
            else:
                print(".", end="", flush=True)
            time.sleep(2)
        
        print(f"\nTimeout waiting for server to be ready after {timeout} seconds")
        return False
    
    def test_connection(self) -> bool:
        """Test if the server is responding"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_container_logs(self, lines: int = 50) -> str:
        """Get container logs"""
        try:
            result = self._run_command(["podman", "logs", "--tail", str(lines), self.container_name])
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error getting logs: {result.stderr}"
        except Exception as e:
            return f"Error getting logs: {e}"
    
    def get_container_status(self) -> Dict:
        """Get detailed container status"""
        try:
            result = self._run_command(["podman", "inspect", self.container_name])
            if result.returncode == 0:
                inspect_data = json.loads(result.stdout)[0]
                return {
                    "name": self.container_name,
                    "status": inspect_data["State"]["Status"],
                    "running": inspect_data["State"]["Running"],
                    "started_at": inspect_data["State"]["StartedAt"],
                    "ports": inspect_data.get("NetworkSettings", {}).get("Ports", {}),
                    "image": inspect_data["Config"]["Image"]
                }
            else:
                return {"error": f"Container not found: {result.stderr}"}
        except Exception as e:
            return {"error": f"Error inspecting container: {e}"}

def main():
    """Test the Podman Manager"""
    print("Podman Manager for Gemma3 Model Serving")
    print("=" * 50)
    
    manager = PodmanManager()
    
    # Check current status
    print(f"Container exists: {manager.is_container_exists()}")
    print(f"Container running: {manager.is_container_running()}")
    
    # Test starting the container
    if input("Start container? (y/N): ").lower() == 'y':
        if manager.start_container():
            print("Container started successfully!")
            
            # Test connection
            if manager.test_connection():
                print("Server is responding!")
            else:
                print("Server is not responding")
                print("Recent logs:")
                print(manager.get_container_logs(20))
        else:
            print("Failed to start container")
    
    # Show status
    status = manager.get_container_status()
    print(f"Container status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    main()