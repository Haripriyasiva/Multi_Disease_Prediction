from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class Encryption:
    def __init__(self, key):
        self.key = key

    def encrypt(self, data):
        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, data.encode('utf-8'), None)
        return nonce + ciphertext

    def decrypt(self, data):
        aesgcm = AESGCM(self.key)
        nonce = data[:12]
        ciphertext = data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None).decode('utf-8')

# Example usage:
# key = AESGCM.generate_key(bit_length=256)
# encryption = Encryption(key)
# encrypted_data = encryption.encrypt("This is a secret message.")
# print(f"Encrypted: {encrypted_data}")
# decrypted_data = encryption.decrypt(encrypted_data)
# print(f"Decrypted: {decrypted_data}")
