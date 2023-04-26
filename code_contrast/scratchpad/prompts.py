from typing import Tuple


def prefix_suffix_sel(
        prefix: str,
        suffix: str = "",
        selection: str = ""
) -> Tuple[str, str, str]:
    return prefix, suffix, selection


def comment_each_line(
        selection: str,
) -> Tuple[str, str, str]:
    prefix = f"""
DO:
-----
Explain the code
-----

CODE:
-----
# language: python
    def completion(self, final: bool, tokens_batch: Optional[int] = 25) -> Iterator[Dict[str, str]]:
        tokens_batch: int = self.max_tokens if final else tokens_batch

        # implement more cool features
        return self.completion_stream(
            # engine must be one of the one in docs
            engine=self._engine,
            tokens_batch=tokens_batch,
            prompt=self.prompt,
            replace_modified=self._replace_modified
        )
-----
ANSWER:
-----
    def completion(self, final: bool, tokens_batch: Optional[int] = 25) -> Iterator[Dict[str, str]]:
        # if not tokens_batch given, using max_tokens
        tokens_batch: int = self.max_tokens if final else tokens_batch

        # implement more cool features
        return self.completion_stream(
            # engine is a model codify API uses. E.g.  text-davinci-003, code-davinci-002 etc
            # engine must be one of the one in docs
            engine=self._engine,
            # how many tokens will be in each batch
            tokens_batch=tokens_batch,
            # function that returns prompt for selected engine
            prompt=self.prompt,
            # replace selection from original code with generated code
            replace_modified=self._replace_modified
        )
-----


DO:
-----
Explain the code
-----
CODE:
-----
# language: python
        # without --all, only master
        stdout = check_output(
            ["git", "-C", str(repo_path), "log", "--format=%H", "--all"],
            timeout=node_subprocess_timeout
        )
-----
ANSWER:
-----
        # creating subprocess to run git log command
        # without --all, only master
        stdout = check_output(
            # execute git log command with custom format and repo_path parameters
            ["git", "-C", str(repo_path), "log", "--format=%H", "--all"],
            # when `timeout` is not set, the command will run forever
            # if the command takes longer than `timeout` seconds to run, exception will be raised
            timeout=node_subprocess_timeout
        )
-----


DO:
-----
Explain the code
-----

CODE:
-----
{selection}
-----
ANSWER:
-----
"""
    return prefix_suffix_sel(prefix=prefix)

