# runtime

本地商业运行时文件放在这里，供 Aspose 加载器读取：

```text
runtime/
  Aspose.Words.dll
  Aspose.PDF.dll
  Aspose.Total.NET.lic
  SkiaSharp.dll
  libSkiaSharp.dll
  DkbgParser.runtimeconfig.json
```

也可以通过 `ASPOSE_DLL_DIR` 和 `ASPOSE_LICENSE_PATH` 指向其他目录。

`DkbgParser.runtimeconfig.json` 会让 pythonnet 使用已安装的 .NET 8 Core、Windows Desktop 和 ASP.NET Core 共享运行时，以满足两套 Aspose DLL 的依赖。
