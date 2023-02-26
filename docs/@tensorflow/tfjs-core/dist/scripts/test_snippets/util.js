/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as fs from 'fs';
import * as path from 'path';
import * as ts from 'typescript';
process.on('unhandledRejection', ex => {
    throw ex;
});
// Used for logging the number of snippets that have been found.
let snippetCount = 0;
// Used for counting the number of errors that have been found.
let errorCount = 0;
/**
 * Parse and evaluate snippets for the src/index.ts from where this script is
 * run.
 * @param tf The TensorFlow.js module to use when evaluating snippets. If used
 *     outside core, this should be a union of core and the separate package.
 *     This is unused here but is used in eval() of the snippets.
 */
// tslint:disable-next-line:no-any
export async function parseAndEvaluateSnippets(tf) {
    const index = path.join(process.cwd(), 'src/index.ts');
    const tsconfigPath = path.join(process.cwd(), 'tsconfig.json');
    // Use the same compiler options that we use to compile the library
    // here.
    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf8'));
    delete tsconfig.compilerOptions.moduleResolution;
    const program = ts.createProgram([index], tsconfig.compilerOptions);
    const checker = program.getTypeChecker();
    for (const sourceFile of program.getSourceFiles()) {
        if (!sourceFile.isDeclarationFile) {
            const children = sourceFile.getChildren();
            for (let i = 0; i < children.length; i++) {
                await visit(tf, checker, children[i], sourceFile);
            }
        }
    }
    if (errorCount === 0) {
        console.log(`Parsed and evaluated ${snippetCount} snippets successfully.`);
    }
    else {
        console.log(`Evaluated ${snippetCount} snippets with ${errorCount} errors.`);
        process.exit(1);
    }
}
async function visit(
// tslint:disable-next-line:no-any
tf, checker, node, sourceFile) {
    const children = node.getChildren();
    for (let i = 0; i < children.length; i++) {
        await visit(tf, checker, children[i], sourceFile);
    }
    if (ts.isClassDeclaration(node) || ts.isFunctionDeclaration(node) ||
        ts.isMethodDeclaration(node) || ts.isInterfaceDeclaration(node)) {
        const symbol = checker.getSymbolAtLocation(node.name);
        const jsdoc = getJSDocTag(symbol);
        if (jsdoc == null) {
            return;
        }
        // Ignore snippets of methods that have been marked with ignoreCI.
        if (jsdoc['ignoreCI']) {
            return;
        }
        const documentation = symbol.getDocumentationComment(checker);
        if (documentation == null) {
            return;
        }
        for (let i = 0; i < documentation.length; i++) {
            const doc = documentation[i];
            const re = /```js.*?```/gs;
            const matches = re.exec(doc.text);
            if (matches == null) {
                return;
            }
            for (let k = 0; k < matches.length; k++) {
                snippetCount++;
                const match = matches[k];
                const lines = match.split('\n');
                const evalLines = [];
                for (let j = 0; j < lines.length; j++) {
                    let line = lines[j];
                    if (line.startsWith('```js')) {
                        line = line.substring('```js'.length);
                    }
                    if (line.endsWith('```')) {
                        line = line.substring(0, line.length - '```'.length);
                    }
                    line = line.trim();
                    if (line.startsWith('*')) {
                        line = line.substring(1).trim();
                    }
                    evalLines.push(line);
                }
                const srcCode = evalLines.join('\n');
                const evalString = '(async function runner() { try { ' + srcCode +
                    '} catch (e) { reportError(e); } })()';
                const oldLog = console.log;
                const oldWarn = console.warn;
                const reportError = (e) => {
                    oldLog();
                    oldLog(`Error executing snippet for ${symbol.name} at ${sourceFile.fileName}`);
                    oldLog();
                    oldLog(`\`\`\`js${srcCode}\`\`\``);
                    oldLog();
                    console.error(e);
                    errorCount++;
                };
                // Overrwrite console.log so we don't spam the console.
                console.log = (msg) => { };
                console.warn = (msg) => { };
                try {
                    await eval(evalString);
                }
                catch (e) {
                    reportError(e);
                }
                console.log = oldLog;
                console.warn = oldWarn;
            }
        }
    }
}
function getJSDocTag(symbol) {
    const tags = symbol.getJsDocTags();
    for (let i = 0; i < tags.length; i++) {
        const jsdocTag = tags[i];
        if (jsdocTag.name === 'doc' && jsdocTag.text != null) {
            if (jsdocTag.text.length !== 1) {
                throw new Error('Expected exactly one jsdoc SymbolDisplayPart but got'
                    + ` ${jsdocTag.text.length} instead: ${jsdocTag.text}`);
            }
            const text = jsdocTag.text[0].text.trim();
            const json = convertDocStringToDocInfoObject(text);
            return json;
        }
    }
    return null;
}
function convertDocStringToDocInfoObject(docString) {
    const jsonString = docString.replace(/([a-zA-Z0-9]+):/g, '"$1":').replace(/\'/g, '"');
    return JSON.parse(jsonString);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zY3JpcHRzL3Rlc3Rfc25pcHBldHMvdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEtBQUssRUFBRSxNQUFNLElBQUksQ0FBQztBQUN6QixPQUFPLEtBQUssSUFBSSxNQUFNLE1BQU0sQ0FBQztBQUM3QixPQUFPLEtBQUssRUFBRSxNQUFNLFlBQVksQ0FBQztBQUVqQyxPQUFPLENBQUMsRUFBRSxDQUFDLG9CQUFvQixFQUFFLEVBQUUsQ0FBQyxFQUFFO0lBQ3BDLE1BQU0sRUFBRSxDQUFDO0FBQ1gsQ0FBQyxDQUFDLENBQUM7QUFFSCxnRUFBZ0U7QUFDaEUsSUFBSSxZQUFZLEdBQUcsQ0FBQyxDQUFDO0FBQ3JCLCtEQUErRDtBQUMvRCxJQUFJLFVBQVUsR0FBRyxDQUFDLENBQUM7QUFFbkI7Ozs7OztHQU1HO0FBQ0gsa0NBQWtDO0FBQ2xDLE1BQU0sQ0FBQyxLQUFLLFVBQVUsd0JBQXdCLENBQUMsRUFBTztJQUNwRCxNQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLEVBQUUsRUFBRSxjQUFjLENBQUMsQ0FBQztJQUN2RCxNQUFNLFlBQVksR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLEVBQUUsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUUvRCxtRUFBbUU7SUFDbkUsUUFBUTtJQUNSLE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLFlBQVksQ0FBQyxZQUFZLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztJQUVuRSxPQUFPLFFBQVEsQ0FBQyxlQUFlLENBQUMsZ0JBQWdCLENBQUM7SUFDakQsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLGFBQWEsQ0FBQyxDQUFDLEtBQUssQ0FBQyxF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LElBQUksRUFBRTtZQUNwRCxJQUFJLFFBQVEsQ0FBQyxJQUFJLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtnQkFDOUIsTUFBTSxJQUFJLEtBQUssQ0FBQyxzREFBc0Q7c0JBQ2xFLElBQUksUUFBUSxDQUFDLElBQUksQ0FBQyxNQUFNLGFBQWEsUUFBUSxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7YUFDM0Q7WUFDRCxNQUFNLElBQUksR0FBRyxRQUFRLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQztZQUMxQyxNQUFNLElBQUksR0FBRywrQkFBK0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUNuRCxPQUFPLElBQUksQ0FBQztTQUNiO0tBQ0Y7SUFDRCxPQUFPLElBQUksQ0FBQztBQUNkLENBQUM7QUFFRCxTQUFTLCtCQUErQixDQUFDLFNBQWlCO0lBQ3hELE1BQU0sVUFBVSxHQUNaLFNBQVMsQ0FBQyxPQUFPLENBQUMsa0JBQWtCLEVBQUUsT0FBTyxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssRUFBRSxHQUFHLENBQUMsQ0FBQztJQUN2RSxPQUFPLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLENBQUM7QUFDaEMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgZnMgZnJvbSAnZnMnO1xuaW1wb3J0ICogYXMgcGF0aCBmcm9tICdwYXRoJztcbmltcG9ydCAqIGFzIHRzIGZyb20gJ3R5cGVzY3JpcHQnO1xuXG5wcm9jZXNzLm9uKCd1bmhhbmRsZWRSZWplY3Rpb24nLCBleCA9PiB7XG4gIHRocm93IGV4O1xufSk7XG5cbi8vIFVzZWQgZm9yIGxvZ2dpbmcgdGhlIG51bWJlciBvZiBzbmlwcGV0cyB0aGF0IGhhdmUgYmVlbiBmb3VuZC5cbmxldCBzbmlwcGV0Q291bnQgPSAwO1xuLy8gVXNlZCBmb3IgY291bnRpbmcgdGhlIG51bWJlciBvZiBlcnJvcnMgdGhhdCBoYXZlIGJlZW4gZm91bmQuXG5sZXQgZXJyb3JDb3VudCA9IDA7XG5cbi8qKlxuICogUGFyc2UgYW5kIGV2YWx1YXRlIHNuaXBwZXRzIGZvciB0aGUgc3JjL2luZGV4LnRzIGZyb20gd2hlcmUgdGhpcyBzY3JpcHQgaXNcbiAqIHJ1bi5cbiAqIEBwYXJhbSB0ZiBUaGUgVGVuc29yRmxvdy5qcyBtb2R1bGUgdG8gdXNlIHdoZW4gZXZhbHVhdGluZyBzbmlwcGV0cy4gSWYgdXNlZFxuICogICAgIG91dHNpZGUgY29yZSwgdGhpcyBzaG91bGQgYmUgYSB1bmlvbiBvZiBjb3JlIGFuZCB0aGUgc2VwYXJhdGUgcGFja2FnZS5cbiAqICAgICBUaGlzIGlzIHVudXNlZCBoZXJlIGJ1dCBpcyB1c2VkIGluIGV2YWwoKSBvZiB0aGUgc25pcHBldHMuXG4gKi9cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBwYXJzZUFuZEV2YWx1YXRlU25pcHBldHModGY6IGFueSkge1xuICBjb25zdCBpbmRleCA9IHBhdGguam9pbihwcm9jZXNzLmN3ZCgpLCAnc3JjL2luZGV4LnRzJyk7XG4gIGNvbnN0IHRzY29uZmlnUGF0aCA9IHBhdGguam9pbihwcm9jZXNzLmN3ZCgpLCAndHNjb25maWcuanNvbicpO1xuXG4gIC8vIFVzZSB0aGUgc2FtZSBjb21waWxlciBvcHRpb25zIHRoYXQgd2UgdXNlIHRvIGNvbXBpbGUgdGhlIGxpYnJhcnlcbiAgLy8gaGVyZS5cbiAgY29uc3QgdHNjb25maWcgPSBKU09OLnBhcnNlKGZzLnJlYWRGaWxlU3luYyh0c2NvbmZpZ1BhdGgsICd1dGY4JykpO1xuXG4gIGRlbGV0ZSB0c2NvbmZpZy5jb21waWxlck9wdGlvbnMubW9kdWxlUmVzb2x1dGlvbjtcbiAgY29uc3QgcHJvZ3JhbSA9IHRzLmNyZWF0ZVByb2dyYW0oW2luZGV4XSwgdHNjb25maWcuY29tcGlsZXJPcHRpb25zKTtcblxuICBjb25zdCBjaGVja2VyID0gcHJvZ3JhbS5nZXRUeXBlQ2hlY2tlcigpO1xuXG4gIGZvciAoY29uc3Qgc291cmNlRmlsZSBvZiBwcm9ncmFtLmdldFNvdXJjZUZpbGVzKCkpIHtcbiAgICBpZiAoIXNvdXJjZUZpbGUuaXNEZWNsYXJhdGlvbkZpbGUpIHtcbiAgICAgIGNvbnN0IGNoaWxkcmVuID0gc291cmNlRmlsZS5nZXRDaGlsZHJlbigpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBjaGlsZHJlbi5sZW5ndGg7IGkrKykge1xuICAgICAgICBhd2FpdCB2aXNpdCh0ZiwgY2hlY2tlciwgY2hpbGRyZW5baV0sIHNvdXJjZUZpbGUpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGlmIChlcnJvckNvdW50ID09PSAwKSB7XG4gICAgY29uc29sZS5sb2coYFBhcnNlZCBhbmQgZXZhbHVhdGVkICR7c25pcHBldENvdW50fSBzbmlwcGV0cyBzdWNjZXNzZnVsbHkuYCk7XG4gIH0gZWxzZSB7XG4gICAgY29uc29sZS5sb2coXG4gICAgICAgIGBFdmFsdWF0ZWQgJHtzbmlwcGV0Q291bnR9IHNuaXBwZXRzIHdpdGggJHtlcnJvckNvdW50fSBlcnJvcnMuYCk7XG4gICAgcHJvY2Vzcy5leGl0KDEpO1xuICB9XG59XG5cbmFzeW5jIGZ1bmN0aW9uIHZpc2l0KFxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICB0ZjogYW55LCBjaGVja2VyOiB0cy5UeXBlQ2hlY2tlciwgbm9kZTogdHMuTm9kZSxcbiAgICBzb3VyY2VGaWxlOiB0cy5Tb3VyY2VGaWxlKSB7XG4gIGNvbnN0IGNoaWxkcmVuID0gbm9kZS5nZXRDaGlsZHJlbigpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGNoaWxkcmVuLmxlbmd0aDsgaSsrKSB7XG4gICAgYXdhaXQgdmlzaXQodGYsIGNoZWNrZXIsIGNoaWxkcmVuW2ldLCBzb3VyY2VGaWxlKTtcbiAgfVxuXG4gIGlmICh0cy5pc0NsYXNzRGVjbGFyYXRpb24obm9kZSkgfHwgdHMuaXNGdW5jdGlvbkRlY2xhcmF0aW9uKG5vZGUpIHx8XG4gICAgICB0cy5pc01ldGhvZERlY2xhcmF0aW9uKG5vZGUpIHx8IHRzLmlzSW50ZXJmYWNlRGVjbGFyYXRpb24obm9kZSkpIHtcbiAgICBjb25zdCBzeW1ib2wgPSBjaGVja2VyLmdldFN5bWJvbEF0TG9jYXRpb24obm9kZS5uYW1lKTtcbiAgICBjb25zdCBqc2RvYyA9IGdldEpTRG9jVGFnKHN5bWJvbCk7XG4gICAgaWYgKGpzZG9jID09IG51bGwpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgLy8gSWdub3JlIHNuaXBwZXRzIG9mIG1ldGhvZHMgdGhhdCBoYXZlIGJlZW4gbWFya2VkIHdpdGggaWdub3JlQ0kuXG4gICAgaWYgKGpzZG9jWydpZ25vcmVDSSddKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgZG9jdW1lbnRhdGlvbiA9IHN5bWJvbC5nZXREb2N1bWVudGF0aW9uQ29tbWVudChjaGVja2VyKTtcbiAgICBpZiAoZG9jdW1lbnRhdGlvbiA9PSBudWxsKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZG9jdW1lbnRhdGlvbi5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgZG9jID0gZG9jdW1lbnRhdGlvbltpXTtcbiAgICAgIGNvbnN0IHJlID0gL2BgYGpzLio/YGBgL2dzO1xuICAgICAgY29uc3QgbWF0Y2hlcyA9IHJlLmV4ZWMoZG9jLnRleHQpO1xuICAgICAgaWYgKG1hdGNoZXMgPT0gbnVsbCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgbWF0Y2hlcy5sZW5ndGg7IGsrKykge1xuICAgICAgICBzbmlwcGV0Q291bnQrKztcblxuICAgICAgICBjb25zdCBtYXRjaCA9IG1hdGNoZXNba107XG4gICAgICAgIGNvbnN0IGxpbmVzID0gbWF0Y2guc3BsaXQoJ1xcbicpO1xuICAgICAgICBjb25zdCBldmFsTGluZXM6IHN0cmluZ1tdID0gW107XG4gICAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgbGluZXMubGVuZ3RoOyBqKyspIHtcbiAgICAgICAgICBsZXQgbGluZSA9IGxpbmVzW2pdO1xuICAgICAgICAgIGlmIChsaW5lLnN0YXJ0c1dpdGgoJ2BgYGpzJykpIHtcbiAgICAgICAgICAgIGxpbmUgPSBsaW5lLnN1YnN0cmluZygnYGBganMnLmxlbmd0aCk7XG4gICAgICAgICAgfVxuICAgICAgICAgIGlmIChsaW5lLmVuZHNXaXRoKCdgYGAnKSkge1xuICAgICAgICAgICAgbGluZSA9IGxpbmUuc3Vic3RyaW5nKDAsIGxpbmUubGVuZ3RoIC0gJ2BgYCcubGVuZ3RoKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgbGluZSA9IGxpbmUudHJpbSgpO1xuICAgICAgICAgIGlmIChsaW5lLnN0YXJ0c1dpdGgoJyonKSkge1xuICAgICAgICAgICAgbGluZSA9IGxpbmUuc3Vic3RyaW5nKDEpLnRyaW0oKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgZXZhbExpbmVzLnB1c2gobGluZSk7XG4gICAgICAgIH1cblxuICAgICAgICBjb25zdCBzcmNDb2RlID0gZXZhbExpbmVzLmpvaW4oJ1xcbicpO1xuXG4gICAgICAgIGNvbnN0IGV2YWxTdHJpbmcgPSAnKGFzeW5jIGZ1bmN0aW9uIHJ1bm5lcigpIHsgdHJ5IHsgJyArIHNyY0NvZGUgK1xuICAgICAgICAgICAgJ30gY2F0Y2ggKGUpIHsgcmVwb3J0RXJyb3IoZSk7IH0gfSkoKSc7XG5cbiAgICAgICAgY29uc3Qgb2xkTG9nID0gY29uc29sZS5sb2c7XG4gICAgICAgIGNvbnN0IG9sZFdhcm4gPSBjb25zb2xlLndhcm47XG5cbiAgICAgICAgY29uc3QgcmVwb3J0RXJyb3IgPSAoZTogc3RyaW5nfEVycm9yKSA9PiB7XG4gICAgICAgICAgb2xkTG9nKCk7XG4gICAgICAgICAgb2xkTG9nKGBFcnJvciBleGVjdXRpbmcgc25pcHBldCBmb3IgJHtzeW1ib2wubmFtZX0gYXQgJHtcbiAgICAgICAgICAgICAgc291cmNlRmlsZS5maWxlTmFtZX1gKTtcbiAgICAgICAgICBvbGRMb2coKTtcbiAgICAgICAgICBvbGRMb2coYFxcYFxcYFxcYGpzJHtzcmNDb2RlfVxcYFxcYFxcYGApO1xuICAgICAgICAgIG9sZExvZygpO1xuXG4gICAgICAgICAgY29uc29sZS5lcnJvcihlKTtcbiAgICAgICAgICBlcnJvckNvdW50Kys7XG4gICAgICAgIH07XG5cbiAgICAgICAgLy8gT3ZlcnJ3cml0ZSBjb25zb2xlLmxvZyBzbyB3ZSBkb24ndCBzcGFtIHRoZSBjb25zb2xlLlxuICAgICAgICBjb25zb2xlLmxvZyA9IChtc2c6IHN0cmluZykgPT4ge307XG4gICAgICAgIGNvbnNvbGUud2FybiA9IChtc2c6IHN0cmluZykgPT4ge307XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgYXdhaXQgZXZhbChldmFsU3RyaW5nKTtcbiAgICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICAgIHJlcG9ydEVycm9yKGUpO1xuICAgICAgICB9XG4gICAgICAgIGNvbnNvbGUubG9nID0gb2xkTG9nO1xuICAgICAgICBjb25zb2xlLndhcm4gPSBvbGRXYXJuO1xuICAgICAgfVxuICAgIH1cbiAgfVxufVxuXG5pbnRlcmZhY2UgSlNEb2Mge1xuICBuYW1lc3BhY2U/OiBzdHJpbmc7XG4gIGlnbm9yZUNJPzogYm9vbGVhbjtcbn1cblxuZnVuY3Rpb24gZ2V0SlNEb2NUYWcoc3ltYm9sOiB0cy5TeW1ib2wpOiBKU0RvYyB7XG4gIGNvbnN0IHRhZ3MgPSBzeW1ib2wuZ2V0SnNEb2NUYWdzKCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgdGFncy5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IGpzZG9jVGFnID0gdGFnc1tpXTtcbiAgICBpZiAoanNkb2NUYWcubmFtZSA9PT0gJ2RvYycgJiYganNkb2NUYWcudGV4dCAhPSBudWxsKSB7XG4gICAgICBpZiAoanNkb2NUYWcudGV4dC5sZW5ndGggIT09IDEpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdFeHBlY3RlZCBleGFjdGx5IG9uZSBqc2RvYyBTeW1ib2xEaXNwbGF5UGFydCBidXQgZ290J1xuICAgICAgICAgICsgYCAke2pzZG9jVGFnLnRleHQubGVuZ3RofSBpbnN0ZWFkOiAke2pzZG9jVGFnLnRleHR9YCk7XG4gICAgICB9XG4gICAgICBjb25zdCB0ZXh0ID0ganNkb2NUYWcudGV4dFswXS50ZXh0LnRyaW0oKTtcbiAgICAgIGNvbnN0IGpzb24gPSBjb252ZXJ0RG9jU3RyaW5nVG9Eb2NJbmZvT2JqZWN0KHRleHQpO1xuICAgICAgcmV0dXJuIGpzb247XG4gICAgfVxuICB9XG4gIHJldHVybiBudWxsO1xufVxuXG5mdW5jdGlvbiBjb252ZXJ0RG9jU3RyaW5nVG9Eb2NJbmZvT2JqZWN0KGRvY1N0cmluZzogc3RyaW5nKTogSlNEb2Mge1xuICBjb25zdCBqc29uU3RyaW5nID1cbiAgICAgIGRvY1N0cmluZy5yZXBsYWNlKC8oW2EtekEtWjAtOV0rKTovZywgJ1wiJDFcIjonKS5yZXBsYWNlKC9cXCcvZywgJ1wiJyk7XG4gIHJldHVybiBKU09OLnBhcnNlKGpzb25TdHJpbmcpO1xufVxuIl19