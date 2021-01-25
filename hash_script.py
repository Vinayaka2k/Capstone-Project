import hashlib

def md5_hash(s, encoding):
	md5_hasher=hashlib.md5()
	md5_hasher.update(s.encode(encoding))
	return md5_hasher.hexdigest()

with open("input.txt") as f:
	for line in f:
		print(md5_hash(line,"utf-8"))