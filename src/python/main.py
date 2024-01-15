from openai import OpenAI
import json
import io
import sys
import warnings
import traceback

warnings.filterwarnings("ignore", message=r"\[W033\]", category=UserWarning)

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

client = OpenAI(api_key="open-api-key")

def main(input_json):
    text = input_json["text"].replace("\n", " ")

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )

    return {
        "embedding": response.data[0].embedding
    }


if __name__ == '__main__':

    input_json = None
    for line in input_stream:

        # read json from stdin
        input_json = json.loads(line)

        try:

            output = main(input_json)
        except BaseException as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info()

            output = {"error": ''}
            output['error'] += "Exception type : %s; \n" % ex_type.__name__
            output['error'] += "Exception message : %s\n" % ex_value
            output['error'] += "Exception traceback : %s\n" % "".join(
                traceback.TracebackException.from_exception(ex).format())

        output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
        sys.stdout.buffer.write(output_json)
        print()