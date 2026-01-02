from typing import Protocol, Any
from webdav3.client import Client


class RemoteStorage(Protocol):
    def send(self, local_file_name: str, remote_file_name: str):
        ...
    def retrieve(self, remote_file_name: str, local_file_name: str):
        ...
    
class WebDavRemoteStorage:
    def __init__(self, webdav_base_uri: str, username: str, passwd: str):
        self.webdav_base_uri = webdav_base_uri
        self.username = username
        self.passwd = passwd
        self.client = Client({
            'webdav_hostname': webdav_base_uri,
            'webdav_login': username,
            'webdav_password': passwd
        })
        
    def send(self, local_file_name: str, remote_file_name: str):
        """Upload a local file to the WebDAV server."""
        from webdav3.urn import Urn
        import os

        # Ensure remote path starts with a leading slash
        if not remote_file_name.startswith('/'):
            remote_file_name = '/' + remote_file_name

        urn = Urn(remote_file_name)

        # Check if parent directory exists, if not try to create it
        parent = urn.parent()
        if parent != '/' and not self.client.check(parent):
            # Try to create parent directories recursively
            parts = [p for p in parent.split('/') if p]
            current_path = ''
            for part in parts:
                current_path += '/' + part
                if not self.client.check(current_path):
                    try:
                        self.client.mkdir(current_path)
                    except:
                        pass  # Directory might already exist or we don't have permissions

        # Now upload the file
        with open(local_file_name, 'rb') as f:
            self.client.execute_request(action='upload', path=urn.quote(), data=f)

    def retrieve(self, remote_file_name: str, local_file_name: str):
        """Download a file from the WebDAV server to a local path."""
        # Ensure remote path starts with a leading slash
        if not remote_file_name.startswith('/'):
            remote_file_name = '/' + remote_file_name
        self.client.download_sync(remote_path=remote_file_name, local_path=local_file_name)


def factory(remote_file_type: str, info: dict[str, Any]) -> RemoteStorage | None:
    if remote_file_type == "webdav":
        server = info.get("server")
        if "webdav.doodledome.org" in server:
            return WebDavRemoteStorage(server, "dmiles", "secret123")
    return None

def main():
    webdav_remote_storage = WebDavRemoteStorage("https://webdav.doodledome.org", "dmiles", "secret123")
    webdav_remote_storage.send("test_data/example.m4a", "example.m4a")
    webdav_remote_storage.retrieve("example.m4a", "/tmp/retrieved_example.m4a")
if __name__ == "__main__":
    main()