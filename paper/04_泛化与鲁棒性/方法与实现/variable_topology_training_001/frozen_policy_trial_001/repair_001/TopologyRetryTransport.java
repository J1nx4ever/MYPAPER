package fusion;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;

/** Retries only transport failures for the same idempotent frozen-policy request. */
public final class TopologyRetryTransport {
    private static final URI ENDPOINT = URI.create("http://127.0.0.1:50871/decide");
    private static final Duration TIMEOUT = Duration.ofSeconds(15);
    private static final int MAX_ATTEMPTS = 3;
    private final HttpClient client = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(TIMEOUT).build();
    private final Path errorLog;

    public TopologyRetryTransport(Path output) {
        errorLog = output.resolve("native-error.log");
    }

    JsonObject post(JsonObject body) {
        String payload = body.toString();
        HttpRequest request = HttpRequest.newBuilder(ENDPOINT).timeout(TIMEOUT)
                .header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofString(payload)).build();
        IOException firstFailure = null;
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                var response = client.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() != 200) {
                    String detail = response.body().strip();
                    if (detail.length() > 1024) detail = detail.substring(0, 1024);
                    throw new IllegalStateException("Frozen policy RPC returned HTTP " + response.statusCode()
                            + " for decision " + body.get("decision_sequence") + " after " + attempt
                            + " attempt(s). Response: " + detail);
                }
                return JsonParser.parseString(response.body()).getAsJsonObject();
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Frozen policy RPC interrupted", error);
            } catch (IOException error) {
                if (firstFailure == null) firstFailure = error;
                else firstFailure.addSuppressed(error);
                if (attempt < MAX_ATTEMPTS) {
                    System.err.printf("[TOPOLOGY_RPC_RETRY] decision=%s attempt=%d cause=%s%n",
                            body.get("decision_sequence"), attempt, error);
                    pauseBeforeRetry(attempt);
                }
            }
        }
        throw new IllegalStateException("Frozen policy RPC unavailable after " + MAX_ATTEMPTS
                + " attempts for decision " + body.get("decision_sequence"), firstFailure);
    }

    private static void pauseBeforeRetry(int attempt) {
        try {
            Thread.sleep(100L * attempt);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Frozen policy RPC retry was interrupted", error);
        }
    }

    public void recordFailure(String phase, long run, long sequence, int truck, double seconds, Throwable error) {
        var stack = new StringWriter();
        error.printStackTrace(new PrintWriter(stack));
        String text = "phase=" + phase + " run_index=" + run + " decision_sequence=" + sequence
                + " truck_id=" + truck + " sim_time_seconds=" + seconds + "\n" + stack;
        try {
            Files.writeString(errorLog, text, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        } catch (FileAlreadyExistsException recorded) {
            return;
        } catch (IOException logError) {
            error.addSuppressed(logError);
        }
        System.err.print("[TOPOLOGY_FIRST_FAILURE] " + text);
    }
}
