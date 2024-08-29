# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
import base64
import hashlib
import logging
import os
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger("encryption")


def encrypt_message(plaintext: str, key: bytes) -> str:
    """Encrypts a message using a key with AES 256 encryption.

    Args:
        plaintext (str): The message to encrypt.
        key (bytes): The encryption key, should be 256 bits.

    Returns:
        str: The encrypted message encoded in base64.
    """
    nounce = os.urandom(12)
    cipher = Cipher(algorithms.AES(key), modes.GCM(nounce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    return base64.b64encode(nounce + ciphertext + encryptor.tag).decode("utf-8")


def decrypt_message(encrypted: str, key: bytes) -> str:
    """Decrypts an encrypted message using a key with AES 256 encryption.

    Args:
        encrypted (str): The encrypted message encoded in base64.
        key (bytes): The encryption key, should be 256 bits.

    Returns:
        str: The decrypted message.
    """
    encrypted = base64.b64decode(encrypted)
    nounce = encrypted[:12]
    ciphertext = encrypted[12:-16]
    tag = encrypted[-16:]
    cipher = Cipher(
        algorithms.AES(key), modes.GCM(nounce, tag), backend=default_backend()
    )
    decryptor = cipher.decryptor()
    return (decryptor.update(ciphertext) + decryptor.finalize()).decode("utf-8")


def generate_key_from_env() -> Optional[bytes]:
    """Generate the encryption key from the environment variable `CRAB_ENC_KEY`.

    Returns:
        Optional[bytes]: The encryption key. If the environment variable is not set or
            empty, return None.
    """
    enc_key = os.environ.get("CRAB_ENC_KEY")
    # don't encrypt as long as the key is an empty value
    if not enc_key:
        logger.warning("CRAB_ENC_KEY is not set, connection will not be encrypted.")
        return None

    return hashlib.sha256(enc_key.encode("utf-8")).digest()
