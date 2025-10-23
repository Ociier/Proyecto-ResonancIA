using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

// URL del backend Flask (puedes sobreescribir con env var FLASK_BASEURL)
var flaskBase = builder.Configuration["Flask:BaseUrl"]
               ?? Environment.GetEnvironmentVariable("FLASK_BASEURL")
               ?? "http://127.0.0.1:5000";

builder.Services.AddRazorPages();
builder.Services.AddHttpClient("flask", c =>
{
    c.BaseAddress = new Uri(flaskBase);
    c.Timeout = TimeSpan.FromMinutes(2);
});

// (opcional) subir archivos grandes
builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 1024L * 1024L * 1024L; // 1GB
});

var app = builder.Build();

app.UseStaticFiles();
app.UseRouting();
app.MapRazorPages();

app.Run();
