from zeep import Client
import hashlib
# 客户端
wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
client = Client(wsdl)
# 用户名和密码
email = "2817919375@qq.com" # 首先在Brenda上用邮箱注册一个账号。
password = hashlib.sha256("wr28170506".encode("utf-8")).hexdigest()
# 获取ecNumber = 1.1.1.1 & organism = Homo sapiens的kmValue的案例。(后详述)
parameters = (email,password,"ecNumber*1.1.1.1","organism*Homo sapiens","kmValue*", "kmValueMaximum*","substrate*","commentary*","ligandStructureId*","literature*" )
resultString = client.service.getKmValue(*parameters)
print (resultString)
