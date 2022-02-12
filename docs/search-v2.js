/*
    This file is part of m.css.

    Copyright © 2017, 2018, 2019, 2020, 2021, 2022
              Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

"use strict"; /* it summons the Cthulhu in a proper way, they say */

var Search = {
    formatVersion: 2, /* the data filename contains this number too */

    dataSize: 0, /* used mainly by tests, not here */
    symbolCount: '&hellip;',
    trie: null,
    map: null,
    mapFlagsOffset: null,
    typeMap: null,
    maxResults: 0,

    /* Type sizes and masks. The data is always fetched as 16/32bit number and
       then masked to 1, 2, 3 or 4 bytes. Fortunately on LE a mask is enough,
       on BE we'd have to read N bytes before and then mask. */
    nameSizeBytes: null,
    nameSizeMask: null,
    resultIdBytes: null,
    resultIdMask: null,
    fileOffsetBytes: null,
    fileOffsetMask: null,
    lookaheadBarrierMask: null,

    /* Always contains at least the root node offset and then one node offset
       per entered character */
    searchString: '',
    searchStack: [],

    /* So items don't get selected right away when a cursor is over results but
       only after mouse moves */
    mouseMovedSinceLastRender: false,

    /* Whether we can go back in history in order to hide the search box or
       not. We can't do that if we arrived directly on #search from outside. */
    canGoBackToHideSearch: false,

    /* Autocompletion in the input field is whitelisted only for character
       input (so not deletion, cut, or anything else). This is flipped in the
       onkeypress event and reset after each oninput event. */
    autocompleteNextInputEvent: false,

    init: function(buffer, maxResults) {
        let view = new DataView(buffer);

        /* The file is too short to contain at least the headers and empty
           sections */
        if(view.byteLength < 31) {
            console.error("Search data too short");
            return false;
        }

        if(view.getUint8(0) != 'M'.charCodeAt(0) ||
           view.getUint8(1) != 'C'.charCodeAt(0) ||
           view.getUint8(2) != 'S'.charCodeAt(0)) {
            console.error("Invalid search data signature");
            return false;
        }

        if(view.getUint8(3) != this.formatVersion) {
            console.error("Invalid search data version");
            return false;
        }

        /* Fetch type sizes. The only value that can fail is result ID byte
           count, where value of 3 has no assigned meaning. */
        let typeSizes = view.getUint8(4, true);
        if((typeSizes & 0x01) >> 0 == 0) {
            this.fileOffsetBytes = 3;
            this.fileOffsetMask = 0x00ffffff;
            this.lookaheadBarrierMask = 0x00800000;
        } else /* (typeSizes & 0x01) >> 0 == 1 */ {
            this.fileOffsetBytes = 4;
            this.fileOffsetMask = 0xffffffff;
            this.lookaheadBarrierMask = 0x80000000;
        }
        if((typeSizes & 0x06) >> 1 == 0) {
            this.resultIdBytes = 2;
            this.resultIdMask = 0x0000ffff;
        } else if((typeSizes & 0x06) >> 1 == 1) {
            this.resultIdBytes = 3;
            this.resultIdMask = 0x00ffffff;
        } else if((typeSizes & 0x06) >> 1 == 2) {
            this.resultIdBytes = 4;
            this.resultIdMask = 0xffffffff;
        } else /* (typeSizes & 0x06) >> 1 == 3 */ {
            console.error("Invalid search data result ID byte value");
            return false;
        }
        if((typeSizes & 0x08) >> 3 == 0) {
            this.nameSizeBytes = 1;
            this.nameSizeMask = 0x00ff;
        } else /* (typeSizes & 0x08) >> 3 == 1 */ {
            this.nameSizeBytes = 2;
            this.nameSizeMask = 0xffff;
        }

        /* Separate the data into the trie and the result / type map. Because
           we're reading larger values than there might be and then masking out
           the high bytes, keep extra 1/2 byte padding at the end to avoid
           OOB errors. */
        let mapOffset = view.getUint32(12, true);
        let typeMapOffset = view.getUint32(16, true);
        /* There may be a 3-byte file offset at the end of the trie which we'll
           read as 32-bit, add one safety byte in that case */
        this.trie = new DataView(buffer, 20, mapOffset - 20 + (4 - this.fileOffsetBytes));
        /* There may be a 3-byte file size (for zero results) which we'll read
           as 32-bit, add one safety byte in that case */
        this.map = new DataView(buffer, mapOffset, typeMapOffset - mapOffset + (4 - this.fileOffsetBytes));
        /* No variable-size types in the type map at the moment */
        this.typeMap = new DataView(buffer, typeMapOffset);

        /* Offset of the first result map item is after N + 1 offsets and N
           flags, calculate flag offset from that */
        this.mapFlagsOffset = this.fileOffsetBytes*(((this.map.getUint32(0, true) & this.fileOffsetMask) - this.fileOffsetBytes)/(this.fileOffsetBytes + 1) + 1);

        /* Set initial properties */
        this.dataSize = buffer.byteLength;
        this.symbolCount = view.getUint32(8, true) + " symbols (" + Math.round(this.dataSize/102.4)/10 + " kB)";
        this.maxResults = maxResults ? maxResults : 100;
        this.searchString = '';
        this.searchStack = [this.trie.getUint32(0, true)];

        /* istanbul ignore if */
        if(typeof document !== 'undefined') {
            document.getElementById('search-symbolcount').innerHTML = this.symbolCount;
            document.getElementById('search-input').disabled = false;
            document.getElementById('search-input').placeholder = "Type something here …";
            document.getElementById('search-input').focus();

            /* Search for the input value (there might be something already,
               for example when going back in the browser) */
            let value = document.getElementById('search-input').value;

            /* Otherwise check the GET parameters for `q` and fill the input
               with that */
            if(!value.length) {
                var args = decodeURIComponent(window.location.search.substr(1)).trim().split('&');
                for(var i = 0; i != args.length; ++i) {
                    if(args[i].substring(0, 2) != 'q=') continue;

                    value = document.getElementById('search-input').value = args[i].substring(2);
                    break;
                }
            }

            if(value.length) Search.searchAndRender(value);
        }

        return true;
    },

    download: /* istanbul ignore next */ function(url) {
        var req = window.XDomainRequest ? new XDomainRequest() : new XMLHttpRequest();
        if(!req) return;

        req.open("GET", url, true);
        req.responseType = 'arraybuffer';
        req.onreadystatechange = function() {
            if(req.readyState != 4) return;

            Search.init(req.response);
        }
        req.send();
    },

    base85decode: function(base85string) {
        function charValue(char) {
            if(char >=  48 && char <  58) /* 0-9 -> 0-9 */
                return char - 48 + 0;
            if(char >=  65 && char <  91) /* A-Z -> 10-35 */
                return char - 65 + 10;
            if(char >=  97 && char < 123) /* a-z -> 36-61 */
                return char - 97 + 36;
            if(char ==  33)               /*  !  -> 62 */
                return 62;
            /* skipping 34 (') */
            if(char >=  35 && char <  39) /* #-& -> 63-66 */
                return char - 35 + 63;
            /* skipping 39 (") */
            if(char >=  40 && char <  44) /* (-+ -> 67-70 */
                return char - 40 + 67;
            /* skipping 44 (,) */
            if(char ==  45)               /*  -  -> 71 */
                return 71;
            if(char >=  59 && char <  65) /* ;-@ -> 72-77 */
                return char - 59 + 72;
            if(char >=  94 && char <  97) /* ^-` -> 78-80 */
                return char - 94 + 78;
            if(char >= 123 && char < 127) /* {-~ -> 81-84 */
                return char - 123 + 81;

            return 0; /* Interpret padding values as zeros */
        }

        /* Pad the string for easier decode later. We don't read past the file
           end, so it doesn't matter what garbage is there. */
        if(base85string.length % 5) {
            console.log("Expected properly padded base85 data");
            return;
        }

        let buffer = new ArrayBuffer(base85string.length*4/5);
        let data8 = new DataView(buffer);
        for(let i = 0; i < base85string.length; i += 5) {
            let char1 = charValue(base85string.charCodeAt(i + 0));
            let char2 = charValue(base85string.charCodeAt(i + 1));
            let char3 = charValue(base85string.charCodeAt(i + 2));
            let char4 = charValue(base85string.charCodeAt(i + 3));
            let char5 = charValue(base85string.charCodeAt(i + 4));

            data8.setUint32(i*4/5, char5 +
                                   char4*85 +
                                   char3*85*85 +
                                   char2*85*85*85 +
                                   char1*85*85*85*85, false); /* BE, yes */
        }

        return buffer;
    },

    load: function(base85string) {
        return this.init(this.base85decode(base85string));
    },

    /* http://ecmanaut.blogspot.com/2006/07/encoding-decoding-utf8-in-javascript.html */
    toUtf8: function(string) { return unescape(encodeURIComponent(string)); },
    fromUtf8: function(string) { return decodeURIComponent(escape(string)); },

    autocompletedCharsToUtf8: function(chars) {
        /* Strip incomplete UTF-8 chars from the autocompletion end */
        for(let i = chars.length - 1; i >= 0; --i) {
            let c = chars[i];

            /* We're safe, finish */
            if(
                /* ASCII value at the end */
                (c < 128 && i + 1 == chars.length) ||

                /* Full two-byte character at the end */
                ((c & 0xe0) == 0xc0 && i + 2 == chars.length) ||

                /* Full three-byte character at the end */
                ((c & 0xf0) == 0xe0 && i + 3 == chars.length) ||

                /* Full four-byte character at the end */
                ((c & 0xf8) == 0xf0 && i + 4 == chars.length)
            ) break;

            /* Continuing UTF-8 character, go further back */
            if((c & 0xc0) == 0x80) continue;

            /* Otherwise the character is not complete, drop it from the end */
            chars.length = i;
            break;
        }

        /* Convert the autocompleted UTF-8 sequence to a string */
        let suggestedTabAutocompletionString = '';
        for(let i = 0; i != chars.length; ++i)
            suggestedTabAutocompletionString += String.fromCharCode(chars[i]);
        return suggestedTabAutocompletionString;
    },

    /* Returns the values in UTF-8, but input is in whatever shitty 16bit
       encoding JS has */
    search: function(searchString) {
        /* Normalize the search string first, convert to UTF-8 and trim spaces
           from the left. From the right they're trimmed only if nothing is
           found, see below. */
        searchString = this.toUtf8(searchString.toLowerCase().replace(/^\s+/,''));

        /* TODO: maybe i could make use of InputEvent.data and others here */

        /* Find longest common prefix of previous and current value so we don't
           need to needlessly search again */
        let max = Math.min(searchString.length, this.searchString.length);
        let commonPrefix = 0;
        for(; commonPrefix != max; ++commonPrefix)
            if(searchString[commonPrefix] != this.searchString[commonPrefix]) break;

        /* Drop items off the stack if it has has more than is needed for the
           common prefix (it needs to have at least one item, though) */
        if(commonPrefix + 1 < this.searchStack.length)
            this.searchStack.splice(commonPrefix + 1, this.searchStack.length - commonPrefix - 1);

        /* Add new characters from the search string */
        let foundPrefix = commonPrefix;
        for(; foundPrefix != searchString.length; ++foundPrefix) {
            /* Calculate offset and count of children */
            let offset = this.searchStack[this.searchStack.length - 1];

            /* If there's a lot of results, the result count is a 16bit BE value
               instead */
            let resultCount = this.trie.getUint8(offset);
            let resultCountSize = 1;
            if(resultCount & 0x80) {
                resultCount = this.trie.getUint16(offset, false) & ~0x8000;
                ++resultCountSize;
            }

            let childCount = this.trie.getUint8(offset + resultCountSize);

            /* Go through all children and find the next offset */
            let childOffset = offset + resultCountSize + 1 + resultCount*this.resultIdBytes;
            let found = false;
            for(let j = 0; j != childCount; ++j) {
                if(String.fromCharCode(this.trie.getUint8(childOffset + j)) != searchString[foundPrefix])
                    continue;

                this.searchStack.push(this.trie.getUint32(childOffset + childCount + j*this.fileOffsetBytes, true) & this.fileOffsetMask & ~this.lookaheadBarrierMask);
                found = true;
                break;
            }

            /* Character not found */
            if(!found) {
                /* If we found everything except spaces at the end, pretend the
                   spaces aren't there. On the other hand, we *do* want to
                   try searching with the spaces first -- it can narrow down
                   the result list for page names or show subpages (which are
                   after a lookahead barrier that's a space). */
                if(!searchString.substr(foundPrefix).trim().length)
                    searchString = searchString.substr(0, foundPrefix);

                break;
            }
        }

        /* Save the whole found prefix for next time */
        this.searchString = searchString.substr(0, foundPrefix);

        /* If the whole thing was not found, return an empty result and offer
           external search */
        if(foundPrefix != searchString.length) {
            /* istanbul ignore if */
            if(typeof document !== 'undefined') {
                let link = document.getElementById('search-external');
                if(link)
                    link.href = link.dataset.searchEngine.replace('{query}', encodeURIComponent(searchString));
            }
            return [[], ''];
        }

        /* Otherwise gather the results */
        let suggestedTabAutocompletionChars = [];
        let results = [];
        let leaves = [[this.searchStack[this.searchStack.length - 1], 0]];
        while(leaves.length) {
            /* Pop offset from the queue */
            let current = leaves.shift();
            let offset = current[0];
            let suffixLength = current[1];

            /* Calculate child count. If there's a lot of results, the count
               "leaks over" to the child count storage. */
            /* TODO: hmmm. this is helluvalot duplicated code. hmm. */
            let resultCount = this.trie.getUint8(offset);
            let resultCountSize = 1;
            if(resultCount & 0x80) {
                resultCount = this.trie.getUint16(offset, false) & ~0x8000;
                ++resultCountSize;
            }

            let childCount = this.trie.getUint8(offset + resultCountSize);

            /* Populate the results with all values associated with this node */
            for(let i = 0; i != resultCount; ++i) {
                let index = this.trie.getUint32(offset + resultCountSize + 1 + i*this.resultIdBytes, true) & this.resultIdMask;
                results.push(this.gatherResult(index, suffixLength, 0xffffff)); /* should be enough haha */

                /* 'nuff said. */
                if(results.length >= this.maxResults)
                    return [results, this.autocompletedCharsToUtf8(suggestedTabAutocompletionChars)];
            }

            /* Dig deeper */
            let childOffset = offset + resultCountSize + 1 + resultCount*this.resultIdBytes;
            for(let j = 0; j != childCount; ++j) {
                let offsetBarrier = this.trie.getUint32(childOffset + childCount + j*this.fileOffsetBytes, true) & this.fileOffsetMask;

                /* Lookahead barrier, don't dig deeper */
                if(offsetBarrier & this.lookaheadBarrierMask) continue;

                /* Append to the queue */
                leaves.push([offsetBarrier & ~this.lookaheadBarrierMask, suffixLength + 1]);

                /* We don't have anything yet and this is the only path
                   forward, add the char to suggested Tab autocompletion. Can't
                   extract it from the leftmost 8 bits of offsetBarrier because
                   that would make it negative, have to load as Uint8 instead.
                   Also can't use String.fromCharCode(), because later doing
                   str.charCodeAt() would give me back UTF-16 values, which is
                   absolutely unwanted when all I want is check for truncated
                   UTF-8. */
                if(!results.length && leaves.length == 1 && childCount == 1)
                    suggestedTabAutocompletionChars.push(this.trie.getUint8(childOffset + j));
            }
        }

        return [results, this.autocompletedCharsToUtf8(suggestedTabAutocompletionChars)];
    },

    gatherResult: function(index, suffixLength, maxUrlPrefix) {
        let flags = this.map.getUint8(this.mapFlagsOffset + index);
        let resultOffset = this.map.getUint32(index*this.fileOffsetBytes, true) & this.fileOffsetMask;

        /* The result is an alias, parse the aliased prefix */
        let aliasedIndex = null;
        if((flags & 0xf0) == 0x00) {
            aliasedIndex = this.map.getUint32(resultOffset, true) & this.resultIdMask;
            resultOffset += this.resultIdBytes;
        }

        /* The result has a prefix, parse that first, recursively */
        let name = '';
        let url = '';
        if(flags & (1 << 3)) {
            let prefixIndex = this.map.getUint32(resultOffset, true) & this.resultIdMask;
            let prefixUrlPrefixLength = Math.min(this.map.getUint16(resultOffset + this.resultIdBytes, true) & this.nameSizeMask, maxUrlPrefix);

            let prefix = this.gatherResult(prefixIndex, 0 /*ignored*/, prefixUrlPrefixLength);
            name = prefix.name;
            url = prefix.url;

            resultOffset += this.resultIdBytes + this.nameSizeBytes;
        }

        /* The result has a suffix, extract its length */
        let resultSuffixLength = 0;
        if(flags & (1 << 0)) {
            resultSuffixLength = this.map.getUint16(resultOffset, true) & this.nameSizeMask;
            resultOffset += this.nameSizeBytes;
        }

        let nextResultOffset = this.map.getUint32((index + 1)*this.fileOffsetBytes, true) & this.fileOffsetMask;

        /* Extract name */
        let j = resultOffset;
        for(; j != nextResultOffset; ++j) {
            let c = this.map.getUint8(j);

            /* End of null-delimited name */
            if(!c) {
                ++j;
                break; /* null-delimited */
            }

            name += String.fromCharCode(c); /* eheh. IS THIS FAST?! */
        }

        /* The result is an alias and we're not deep inside resolving a prefix,
           extract the aliased name and URL */
        /* TODO: this abuses 0xffffff to guess how the call stack is deep and
           that's just wrong, fix! */
        if(aliasedIndex != null && maxUrlPrefix == 0xffffff) {
            let alias = this.gatherResult(aliasedIndex, 0 /* ignored */, 0xffffff); /* should be enough haha */

            /* Keeping in UTF-8, as we need that for proper slicing (and concatenating) */
            return {name: name,
                    alias: alias.name,
                    url: alias.url,
                    flags: alias.flags,
                    cssClass: alias.cssClass,
                    typeName: alias.typeName,
                    suffixLength: suffixLength + resultSuffixLength};
        }

        /* Otherwise extract URL from here */
        let max = Math.min(j + maxUrlPrefix - url.length, nextResultOffset);
        for(; j != max; ++j) {
            url += String.fromCharCode(this.map.getUint8(j));
        }

        /* This is an alias, return what we have, without parsed CSS class and
           type name as those are retrieved from the final target type */
        if(!(flags >> 4))
            return {name: name,
                    url: url,
                    flags: flags & 0x0f,
                    suffixLength: suffixLength + resultSuffixLength};

        /* Otherwise, get CSS class and type name for the result label */
        let typeMapIndex = (flags >> 4) - 1;
        let cssClass = [
            /* Keep in sync with _search.py */
            'm-default',
            'm-primary',
            'm-success',
            'm-warning',
            'm-danger',
            'm-info',
            'm-dim'
        ][this.typeMap.getUint8(typeMapIndex*2)];
        let typeNameOffset = this.typeMap.getUint8(typeMapIndex*2 + 1);
        let nextTypeNameOffset = this.typeMap.getUint8((typeMapIndex + 1)*2 + 1);
        let typeName = '';
        for(let j = typeNameOffset; j != nextTypeNameOffset; ++j)
            typeName += String.fromCharCode(this.typeMap.getUint8(j));

        /* Keeping in UTF-8, as we need that for proper slicing (and
           concatenating). Strip the type from the flags, as it's now expressed
           directly. */
        return {name: name,
                url: url,
                flags: flags & 0x0f,
                cssClass: cssClass,
                typeName: typeName,
                suffixLength: suffixLength + resultSuffixLength};
    },

    escape: function(name) {
        return name.replace(/[\"&<>]/g, function (a) {
            return { '"': '&quot;', '&': '&amp;', '<': '&lt;', '>': '&gt;' }[a];
        });
    },
    escapeForRtl: function(name) {
        /* Besides the obvious escaping of HTML entities we also need
           to escape punctuation, because due to the RTL hack to cut
           text off on left side the punctuation characters get
           reordered (of course). Prepending &lrm; works for most
           characters, parentheses we need to *soak* in it. But only
           the right ones. And that for some reason needs to be also for &.
           Huh. https://en.wikipedia.org/wiki/Right-to-left_mark */
        return this.escape(name).replace(/[:=]/g, '&lrm;$&').replace(/(\)|&gt;|&amp;|\/)/g, '&lrm;$&&lrm;');
    },

    renderResults: /* istanbul ignore next */ function(resultsSuggestedTabAutocompletion) {
        if(!this.searchString.length) {
            document.getElementById('search-help').style.display = 'block';
            document.getElementById('search-results').style.display = 'none';
            document.getElementById('search-notfound').style.display = 'none';
            return;
        }

        document.getElementById('search-help').style.display = 'none';

        /* Results found */
        if(resultsSuggestedTabAutocompletion[0].length) {
            let results = resultsSuggestedTabAutocompletion[0];

            document.getElementById('search-results').style.display = 'block';
            document.getElementById('search-notfound').style.display = 'none';

            let list = '';
            for(let i = 0; i != results.length; ++i) {
                /* Labels + */
                list += '<li' + (i ? '' : ' id="search-current"') + '><a href="' + results[i].url + '" onmouseover="selectResult(event)" data-md-link-title="' + this.escape(results[i].name.substr(results[i].name.length - this.searchString.length - results[i].suffixLength)) + '"><div class="m-label m-flat ' + results[i].cssClass + '">' + results[i].typeName + '</div>' + (results[i].flags & 2 ? '<div class="m-label m-danger">deprecated</div>' : '') + (results[i].flags & 4 ? '<div class="m-label m-danger">deleted</div>' : '');

                /* Render the alias (cut off from the right) */
                if(results[i].alias) {
                    list += '<div class="m-doc-search-alias"><span class="m-text m-dim">' + this.escape(results[i].name.substr(0, results[i].name.length - this.searchString.length - results[i].suffixLength)) + '</span><span class="m-doc-search-typed">' + this.escape(results[i].name.substr(results[i].name.length - this.searchString.length - results[i].suffixLength, this.searchString.length)) + '</span>' + this.escapeForRtl(results[i].name.substr(results[i].name.length - results[i].suffixLength)) + '<span class="m-text m-dim">: ' + this.escape(results[i].alias) + '</span>';

                /* Render the normal thing (cut off from the left, have to
                   escape for RTL) */
                } else {
                    list += '<div><span class="m-text m-dim">' + this.escapeForRtl(results[i].name.substr(0, results[i].name.length - this.searchString.length - results[i].suffixLength)) + '</span><span class="m-doc-search-typed">' + this.escapeForRtl(results[i].name.substr(results[i].name.length - this.searchString.length - results[i].suffixLength, this.searchString.length)) + '</span>' + this.escapeForRtl(results[i].name.substr(results[i].name.length - results[i].suffixLength));
                }

                /* The closing */
                list += '</div></a></li>';
            }
            document.getElementById('search-results').innerHTML = this.fromUtf8(list);
            document.getElementById('search-current').scrollIntoView(true);

            /* Append the suggested tab autocompletion, if any, and if the user
               didn't just delete it */
            let searchInput = document.getElementById('search-input');
            if(this.autocompleteNextInputEvent && resultsSuggestedTabAutocompletion[1].length && searchInput.selectionEnd == searchInput.value.length) {
                let suggestedTabAutocompletion = this.fromUtf8(resultsSuggestedTabAutocompletion[1]);

                let lengthBefore = searchInput.value.length;
                searchInput.value += suggestedTabAutocompletion;
                searchInput.setSelectionRange(lengthBefore, searchInput.value.length);
            }

        /* Nothing found */
        } else {
            document.getElementById('search-results').innerHTML = '';
            document.getElementById('search-results').style.display = 'none';
            document.getElementById('search-notfound').style.display = 'block';
        }

        /* Don't allow things to be selected just by motionless mouse cursor
           suddenly appearing over a search result */
        this.mouseMovedSinceLastRender = false;

        /* Reset autocompletion, if it was allowed. It'll get whitelisted next
           time a character gets inserted. */
        this.autocompleteNextInputEvent = false;
    },

    searchAndRender: /* istanbul ignore next */ function(value) {
        let prev = performance.now();
        let results = this.search(value);
        let after = performance.now();
        this.renderResults(results);
        if(this.searchString.length) {
            document.getElementById('search-symbolcount').innerHTML =
                results[0].length + (results[0].length >= this.maxResults ? '+' : '') + " results (" + Math.round((after - prev)*10)/10 + " ms)";
        } else
            document.getElementById('search-symbolcount').innerHTML = this.symbolCount;
    },
};

/* istanbul ignore next */
function selectResult(event) {
    if(!Search.mouseMovedSinceLastRender) return;

    if(event.currentTarget.parentNode.id == 'search-current') return;

    let current = document.getElementById('search-current');
    current.removeAttribute('id');
    event.currentTarget.parentNode.id = 'search-current';
}

/* This is separated from showSearch() because we need non-destructive behavior
   when appearing directly on a URL with #search */ /* istanbul ignore next */
function updateForSearchVisible() {
    /* Prevent accidental scrolling of the body, prevent page layout jumps */
    let scrolledBodyWidth = document.body.offsetWidth;
    document.body.style.overflow = 'hidden';
    document.body.style.paddingRight = (document.body.offsetWidth - scrolledBodyWidth) + 'px';

    document.getElementById('search-input').value = '';
    document.getElementById('search-input').focus();
    document.getElementById('search-results').style.display = 'none';
    document.getElementById('search-notfound').style.display = 'none';
    document.getElementById('search-help').style.display = 'block';
}

/* istanbul ignore next */
function showSearch() {
    window.location.hash = '#search';
    Search.canGoBackToHideSearch = true;

    updateForSearchVisible();
    document.getElementById('search-symbolcount').innerHTML = Search.symbolCount;
    return false;
}

/* istanbul ignore next */
function hideSearch() {
    /* If the search box was opened using showSearch(), we can go back in the
       history. Otherwise (for example when we landed to #search from a
       bookmark or another server), going back would not do the right thing and
       in that case we simply replace the current history state. */
    if(Search.canGoBackToHideSearch) {
        Search.canGoBackToHideSearch = false;
        window.history.back();
    } else {
        window.location.hash = '#!';
        window.history.replaceState('', '', window.location.pathname);
    }

    /* Restore scrollbar, prevent page layout jumps */
    document.body.style.overflow = 'auto';
    document.body.style.paddingRight = '0';

    return false;
}

/* istanbul ignore next */
function copyToKeyboard(text) {
    /* Append to the popup, appending to document.body would cause it to
       scroll when focused */
    let searchPopup = document.getElementsByClassName('m-doc-search')[0];
    let textarea = document.createElement("textarea");
    textarea.value = text;
    searchPopup.appendChild(textarea);
    textarea.focus();
    textarea.select();

    document.execCommand('copy');

    searchPopup.removeChild(textarea);
    document.getElementById('search-input').focus();
}

/* Only in case we're running in a browser. Why a simple if(document) doesn't
   work is beyond me. */ /* istanbul ignore if */
if(typeof document !== 'undefined') {
    document.getElementById('search-input').oninput = function(event) {
        Search.searchAndRender(document.getElementById('search-input').value);
    };

    document.onkeydown = function(event) {
        /* Search shown */
        if(window.location.hash == '#search') {
            /* Close the search */
            if(event.key == 'Escape') {
                hideSearch();

            /* Focus the search input, if not already, using T or Tab */
            } else if((!document.activeElement || document.activeElement.id != 'search-input') && (event.key.toLowerCase() == 't' || event.key == 'Tab') && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
                document.getElementById('search-input').focus();
                return false; /* so T doesn't get entered into the box */

            /* Fill in the autocompleted selection */
            } else if(event.key == 'Tab' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
                /* But only if the input has selection at the end */
                let input = document.getElementById('search-input');
                if(input.selectionEnd == input.value.length && input.selectionStart != input.selectionEnd) {
                    input.setSelectionRange(input.value.length, input.value.length);
                    return false; /* so input won't lose focus */
                }

            /* Select next item */
            } else if(event.key == 'ArrowDown') {
                let current = document.getElementById('search-current');
                if(current) {
                    let next = current.nextSibling;
                    if(next) {
                        current.id = '';
                        next.id = 'search-current';
                        next.scrollIntoView(false);
                    }
                }
                return false; /* so the keypress doesn't affect input cursor */

            /* Select prev item */
            } else if(event.key == 'ArrowUp') {
                let current = document.getElementById('search-current');
                if(current) {
                    let prev = current.previousSibling;
                    if(prev) {
                        current.id = '';
                        prev.id = 'search-current';
                        prev.scrollIntoView(false);
                    }
                }
                return false; /* so the keypress doesn't affect input cursor */

            /* Go to result (if any) */
            } else if(event.key == 'Enter') {
                let result = document.getElementById('search-current');
                if(result) {
                    result.firstElementChild.click();

                    /* We might be staying on the same page, so restore scrollbar,
                       and prevent page layout jumps */
                    document.body.style.overflow = 'auto';
                    document.body.style.paddingRight = '0';
                }
                return false; /* so the form doesn't get sent */

            /* Copy (Markdown) link to keyboard */
            } else if((event.key.toLowerCase() == 'l' || event.key.toLowerCase() == 'm') && event.metaKey) {
                let result = document.getElementById('search-current');
                if(result) {
                    let plain = event.key.toLowerCase() == 'l';
                    let link = plain ? result.firstElementChild.href :
                        '[' + result.firstElementChild.dataset.mdLinkTitle + '](' + result.firstElementChild.href + ')';

                    copyToKeyboard(link);

                    /* Add CSS class to the element for visual feedback (this
                       will get removed on keyup), but only if it's not already
                       there (in case of key repeat, e.g.) */
                    if(result.className.indexOf('m-doc-search-copied') == -1)
                        result.className += ' m-doc-search-copied';
                    console.log("Copied " +  (plain ? "link" : "Markdown link") + " to " + result.firstElementChild.dataset.mdLinkTitle);
                }

                return false; /* so L doesn't get entered into the box */

            /* Looks like the user is inserting some text (and not cutting,
               copying or whatever), allow autocompletion for the new
               character. The oninput event resets this back to false, so this
               basically whitelists only keyboard input, including Shift-key
               and special chars using right Alt (or equivalent on Mac), but
               excluding Ctrl-key, which is usually not for text input. In the
               worst case the autocompletion won't be allowed ever, which is
               much more acceptable behavior than having no ability to disable
               it and annoying the users. */
            } else if(event.key != 'Backspace' && event.key != 'Delete' && !event.metaKey && (!event.ctrlKey || event.altKey)
                /* Don't ever attempt autocompletion with Android virtual
                   keyboards, as those report all `event.key`s as
                   `Unidentified` (on Chrome) or `Process` (on Firefox) with
                   `event.code` 229 and thus we have no way to tell if a text
                   is entered or deleted. See this WONTFIX bug for details:
                    https://bugs.chromium.org/p/chromium/issues/detail?id=118639
                   Couldn't find any similar bugreport for Firefox, but I
                   assume the virtual keyboard is to blame.

                   An alternative is to hook into inputEvent, which has the
                   data, but ... there's more cursed issues right after that:

                    - setSelectionRange() in Chrome on Android only renders
                      stuff, but doesn't actually act as such. Pressing
                      Backspace will only remove the highlight, but the text
                      stays here. Only delay-calling it through a timeout will
                      work as intended. Possibly related SO suggestion (back
                      then not even the rendering worked properly):
                       https://stackoverflow.com/a/13235951
                      Possibly related Chrome bug:
                       https://bugs.chromium.org/p/chromium/issues/detail?id=32865

                    - On Firefox Mobile, programmatically changing an input
                      value (for the autocompletion highlight) will trigger an
                      input event, leading to search *and* autocompletion being
                      triggered again. Ultimately that results in newly typed
                      characters not replacing the autocompletion but rather
                      inserting before it, corrupting the searched string. This
                      event has to be explicitly ignored.

                    - On Firefox Mobile, deleting a highlight with the
                      backspace key will result in *three* input events instead
                      of one:
                        1. `deleteContentBackward` removing the selection (same
                           as Chrome or desktop Firefox)
                        2. `deleteContentBackward` removing *the whole word*
                           that contained the selection (or the whole text if
                           it's just one word)
                        3. `insertCompositionText`, adding the word back in,
                           resulting in the same state as (1).
                      I have no idea WHY it has to do this (possibly some
                      REALLY NASTY workaround to trigger correct font shaping?)
                      but ultimately it results in the autocompletion being
                      added again right after it got deleted, making this whole
                      thing VERY annoying to use.

                   I attempted to work around the above, but it resulted in a
                   huge amount of browser-specific code that achieves only 90%
                   of the goal, with certain corner cases still being rather
                   broken (such as autocompletion randomly triggering when
                   erasing the text, even though it shouldn't). So disabling
                   autocompletion on this HELLISH BROKEN PLATFORM is the best
                   option at the moment. */
                && event.key != 'Unidentified' && event.key != 'Process'
            ) {
                Search.autocompleteNextInputEvent = true;
            /* Otherwise reset the flag, because when the user would press e.g.
               the 'a' key and then e.g. ArrowRight (which doesn't trigger
               oninput), a Backspace after would still result in
               autocompleteNextInputEvent, because nothing reset it back. */
            } else {
                Search.autocompleteNextInputEvent = false;
            }

        /* Search hidden */
        } else {
            /* Open the search on the T or Tab key */
            if((event.key.toLowerCase() == 't' || event.key == 'Tab') && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
                showSearch();
                return false; /* so T doesn't get entered into the box */
            }
        }
    };

    document.onkeyup = function(event) {
        /* Remove highlight after key is released after a link copy */
        if((event.key.toLowerCase() == 'l' || event.key.toLowerCase() == 'm') && event.metaKey) {
            let result = document.getElementById('search-current');
            if(result) result.className = result.className.replace(' m-doc-search-copied', '');
        }
    };

    /* Allow selecting items by mouse hover only after it moves once the
       results are populated. This prevents a random item getting selected if
       the cursor is left motionless over the result area. */
    document.getElementById('search-results').onmousemove = function() {
        Search.mouseMovedSinceLastRender = true;
    };

    /* If #search is already present in the URL, hide the scrollbar etc. for a
       consistent experience */
    if(window.location.hash == '#search') updateForSearchVisible();
}

/* For Node.js testing */ /* istanbul ignore else */
if(typeof module !== 'undefined') { module.exports = { Search: Search }; }
