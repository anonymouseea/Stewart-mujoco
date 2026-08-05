#include "stdafx.h"
#include <windows.h>
#include "zmotion.h"
#include "zauxdll2.h"
void commandCheckHandler(const char *command, int ret)
{
	if (ret)//非 0 则失败
	{
		printf("%s return code is %d\n", command, ret);
    }
}

int _tmain(int argc, _TCHAR* argv[])
{
    char *ip_addr = (char *)"127.0.0.1"; //控制器出厂默认 IP 地址
    192.168.0.11,仿真器 IP 地址 127.0.0.1，仿真器连接需把仿真器打开
    ZMC_HANDLE handle = NULL; //连接句柄
    int ret = ZAux_OpenEth(ip_addr, &handle); //连接控制器
    if (ERR_SUCCESS != ret)
{
printf("控制器连接失败！\n");
handle = NULL;
getchar();
return -1;
}
printf("控制器连接成功！\n");
char SoftTypes[20];
char SoftVersions[20];
char ControllerIDs[20];
ret = ZAux_GetControllerInfo(handle,SoftTypes,SoftVersions,ControllerIDs);//控制卡
信息获取
commandCheckHandler("ZAux_GetControllerInfo", ret) ;//判断指令是否执行成功
printf("SoftType = %s\n", SoftTypes); //打印控制器的型号
printf("SoftVersion = %s\n", SoftVersions); //打印软件版本号