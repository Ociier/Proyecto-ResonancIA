using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Net.Http.Headers;
using System.Text.Json;

public class UploadModel : PageModel
{
    private readonly IHttpClientFactory _httpClientFactory;

    public UploadModel(IHttpClientFactory httpClientFactory)
    {
        _httpClientFactory = httpClientFactory;
    }

    [BindProperty]
    public IFormFile[]? Files { get; set; }

    public List<PredictionResult>? Results { get; set; }
    public string? Error { get; set; }

    public void OnGet() { }

    public async Task<IActionResult> OnPostAsync()
    {
        if (Files == null || Files.Length == 0)
        {
            Error = "Selecciona al menos un archivo (.dcm, .nii, .nii.gz, .tga, .png, .jpg, .jpeg).";
            return Page();
        }

        var client = _httpClientFactory.CreateClient("flask");
        Results = new List<PredictionResult>();

        try
        {
            foreach (var file in Files)
            {
                if (file.Length <= 0) continue;

                using var content = new MultipartFormDataContent();
                using var stream = file.OpenReadStream();
                var streamContent = new StreamContent(stream);

                // Tipo genérico: Flask no lo exige, pero no estorba
                streamContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

                // OJO: el nombre del campo DEBE ser "file" porque Flask usa request.files["file"]
                content.Add(streamContent, "file", file.FileName);

                var resp = await client.PostAsync("/upload", content);
                var json = await resp.Content.ReadAsStringAsync();

                using var doc = JsonDocument.Parse(json);

                if (resp.IsSuccessStatusCode &&
                    doc.RootElement.TryGetProperty("prediction", out var pred))
                {
                    Results.Add(new PredictionResult
                    {
                        FileName = file.FileName,
                        Prediction = pred.GetString() ?? "(sin dato)"
                    });
                }
                else
                {
                    var msg = doc.RootElement.TryGetProperty("error", out var err)
                        ? err.GetString()
                        : $"HTTP {(int)resp.StatusCode}";
                    Results.Add(new PredictionResult
                    {
                        FileName = file.FileName,
                        Prediction = $"Error: {msg}"
                    });
                }
            }
        }
        catch (Exception ex)
        {
            Error = ex.Message;
        }

        return Page();
    }

    public class PredictionResult
    {
        public string FileName { get; set; } = "";
        public string Prediction { get; set; } = "";
    }
}
