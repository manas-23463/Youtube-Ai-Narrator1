import modal

app = modal.App("test-app")

@app.function()
def hello():
    return "Hello from Modal!"

@app.local_entrypoint()
def main():
    print("Testing Modal deployment...")
    result = hello.remote()
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 