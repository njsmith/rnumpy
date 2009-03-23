import rpy2.rinterface
try:
    import njs.rnumpy as rnumpy
except ImportError:
    import rnumpy
import IPython.ipapi
import IPython.hooks
from IPython.genutils import page

MAGIC_WORD = "R"

ip = IPython.ipapi.get()

rnumpy.set_interactive(True)

persistent_R_mode = False
expression_so_far = []

def switch_mode(self, new_mode):
    global persistent_R_mode
    new_mode = not persistent_R_mode
    print ("%s mode (use 'R' or 'R %s' to switch back)."
           % (["Returned to Python", "Entered R"][new_mode],
              ["on", "off"][new_mode]))
    if new_mode:
        print "Tab completion enabled."
    persistent_R_mode = new_mode
    return ""

def rpy_prefilter(self, line):
    global expression_so_far
    # self.buffer is empty iff this is a new (not a continuation) line.
    # It's possible (e.g. if the user hits control-C) that entry could get
    # reset without being completed, and we get no notification if this
    # occurs. So rather than resetting our input buffer whenever it is
    # complete, we reset it every time we see a new line:
    if not self.buffer:
        expression_so_far = []
    line += "\n"
    # Handle persistent toggle:
    if line.strip() == MAGIC_WORD:
        return switch_mode(self, not persistent_R_mode)
    if line.strip().split() == [MAGIC_WORD, "on"]:
        return switch_mode(self, True)
    if line.strip().split() == [MAGIC_WORD, "off"]:
        return switch_mode(self, False)
    # Handle real input:
    if (not self.buffer
        and (persistent_R_mode or line.startswith(MAGIC_WORD + " "))):
        # New line of input
        # Make sure that r evaluation is available in the user namespace:
        self.user_ns["_reval"] = rnumpy.r.__call__
        if not persistent_R_mode:
            # Chop off the magic word at the start:
            line = line[len(MAGIC_WORD):].strip()
        # Convert into a call to _reval:
        if rnumpy.is_complete_expression(line):
            return "_reval(%r)" % (line,)
        elif not self.rc.multi_line_specials:
            # Incomplete expression, but we will not get a chance to fix-up
            # the later lines, so don't even start:
            print "Error: incomplete R expression"
            print "enable rc.multi_line_specials for entry of multi-line R expressions"
            return ""
        else:
            expression_so_far.append(line )
            return "_reval(%r" % (line,)
    elif expression_so_far:
        # This is a continuation line from an earlier magic R expression:
        expression_so_far.append(line)
        if rnumpy.is_complete_expression("".join(expression_so_far)):
            # Finally done: close off the initial _reval( call
            return "+ %r)" % (line,)
        else:
            # Not done yet: append this string and keep going
            return "+ %r" % (line,)
    else:
        raise IPython.ipapi.TryNext

def rpy_complete(self, event):
    global expression_so_far
    if expression_so_far or persistent_R_mode or event.command == MAGIC_WORD:
        cursor = self.Completer.get_endidx()
        line = event.line
        completions = rnumpy.complete(line, cursor)
        return [event.symbol + completion for completion in completions]
    raise IPython.ipapi.TryNext

def rpy_result_display(self, obj):
    # Show R help objects in pager:
    if isinstance(obj, rnumpy.RWrapper):
        try:
            is_help = obj.r.class_()[0] == "help_files_with_topic"
        except:
            is_help = False
        if is_help:
            if len(obj) > 0:
                page(open(obj[0]).read())
            else:
                print "No documentation found for '%s'" % (obj.r.topic()[0],)
            return None
        else:
            raise IPython.ipapi.TryNext
    # When in R mode, display even arrays in "R style":
    elif isinstance(obj, rnumpy.RArray) and persistent_R_mode:
        # Call default:
        IPython.hooks.result_display(self, obj.r)
        return None
    else:
        raise IPython.ipapi.TryNext

def rpy_generate_prompt(self, is_continuation):
    if not persistent_R_mode:
        raise IPython.ipapi.TryNext
    if is_continuation:
        default = str(self.outputcache.prompt2)
    else:
        default = str(self.outputcache.prompt1)
    return default.replace(":", " R>")

def rpy_generate_output_prompt(self):
    if not persistent_R_mode:
        raise IPython.ipapi.TryNext
    default = str(self.outputcache.prompt_out)
    return default.replace(":", " R>")

# Give these high priority, since when they are in effect basically all the
# assumptions that other hooks might make are invalid:
ip.set_hook("input_prefilter", rpy_prefilter, priority=10)
ip.set_hook("complete_command", rpy_complete, priority=10, re_key=".*")
ip.set_hook("result_display", rpy_result_display)
ip.set_hook("generate_prompt", rpy_generate_prompt)
ip.set_hook("generate_output_prompt", rpy_generate_output_prompt)
