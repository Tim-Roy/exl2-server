# import fire
# import os
# import subprocess



# def start_api_server(model: str, host: str="0.0.0.0", port: int=8000):
#     os.environ["EXL2_MODEL"] = model
#     p = subprocess.Popen('ls', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     for line in p.stdout.readlines():
#         print line,
#     retval = p.wait()
#     main:app --host 0.0.0.0 --port 8003 --access-log
#     process = subprocess.Popen(['your_command', 'arg1', 'arg2'],
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE,
#                            text=True)


# if __name__ == "__main__":
#     fire.Fire(start_api_server)
# #