from api import create_app

app = create_app('production')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)
