"""Conservative integer spelling equivalence for data review, never an eval rewrite.

Only a fully equivalent pair may be exempted. Mixed lexical errors, changed values,
decimal/identifier/unsupported formats remain review cases, not automatic bad labels.
"""
import re
import unicodedata

from voxft.eval import _norm

ID = 'nol satu dua tiga empat lima enam tujuh delapan sembilan'.split()
EN = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TH = ('ศูนย์','หนึ่ง','สอง','สาม','สี่','ห้า','หก','เจ็ด','แปด','เก้า')
TL = ('sero','isa','dalawa','tatlo','apat','lima','anim','pito','walo','siyam')
ZH = '零一二三四五六七八九'


def integer_words(n, lang):
    if not 0 <= n < 1_000_000:
        return None
    if lang == 'id':
        if n < 10:return ID[n]
        if n == 10:return 'sepuluh'
        if n == 11:return 'sebelas'
        if n < 20:return ID[n-10]+' belas'
        for base, unit in ((1000,'ribu'),(100,'ratus'),(10,'puluh')):
            if n >= base:
                whole, rest = divmod(n,base)
                prefix = 'se'+unit if whole==1 and base>=100 else integer_words(whole,lang)+' '+unit
                return prefix+(' '+integer_words(rest,lang) if rest else '')
    if lang == 'en':
        if n < 20:return EN[n]
        if n < 100:
            tens=('','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety')
            return tens[n//10]+('-'+EN[n%10] if n%10 else '')
        base, unit = (1000,'thousand') if n>=1000 else (100,'hundred')
        whole, rest=divmod(n,base)
        return integer_words(whole,lang)+' '+unit+(' '+integer_words(rest,lang) if rest else '')
    if lang == 'th':
        if n < 10:return TH[n]
        text=''
        for base,unit in ((100000,'แสน'),(10000,'หมื่น'),(1000,'พัน'),(100,'ร้อย'),(10,'สิบ')):
            digit,n=divmod(n,base)
            if digit:
                text+=('' if base==10 and digit==1 else 'ยี่' if base==10 and digit==2 else TH[digit])+unit
        return text+('เอ็ด' if n==1 else TH[n] if n else '')
    if lang == 'zh':
        if n>=10000:
            whole,rest=divmod(n,10000)
            return integer_words(whole,lang)+'万'+(('零' if rest<1000 else '')+integer_words(rest,lang) if rest else '')
        if n < 10:return ZH[n]
        text=''
        for base,unit in ((1000,'千'),(100,'百'),(10,'十')):
            digit,rest=divmod(n,base)
            if digit:
                text+=('' if base==10 and digit==1 and not text else ZH[digit])+unit
                if 0 < rest < base//10:text+='零'
            n=rest
        return text+(ZH[n] if n else '')
    if lang == 'tl' and n < 100:
        if n < 10:return TL[n]
        if n == 10:return 'sampu'
        if n < 20:return ('labing-isa','labindalawa','labintatlo','labing-apat','labinlima',
                         'labing-anim','labimpito','labingwalo','labinsiyam')[n-11]
        text=('','','dalawampu','tatlumpu','apatnapu','limampu','animnapu','pitumpu','walumpu','siyamnapu')[n//10]
        return text+("'t "+TL[n%10] if n%10 else '')
    return None


def aliases(n, lang):
    if isinstance(n,tuple):
        _,hour,minute=n
        hours={str(hour),f'{hour:02d}',integer_words(hour,'zh')}
        minutes={str(minute),f'{minute:02d}',integer_words(minute,'zh')}
        if minute<10:minutes.add('零'+integer_words(minute,'zh'))
        return sorted({h+'点'+m+'分' for h in hours for m in minutes}
                      |{h+':'+f'{minute:02d}' for h in hours if h.isascii()}
                      |{h+'：'+f'{minute:02d}' for h in hours if h.isascii()})
    word=integer_words(n,lang)
    if word is None:return []
    forms={word}
    if lang=='en' and 'hundred ' in word:
        forms.add(word.replace('hundred ','hundred and '))
    if lang=='zh':
        forms.add(word.replace('二万','两万').replace('二千','两千').replace('二百','两百'))
    if lang=='tl':
        if n==17:forms.add('labing pito')
        # The linker attaches to a numeral before a following noun.
        forms|={s+'ng' for s in forms if s[-1] in 'aeiou'}
        if n==6:forms.add('anim na')
    return sorted(forms,key=lambda s:(-len(s),s))


def _digits(text, lang):
    spans=[]
    unresolved=False
    if lang=='zh':
        for match in re.finditer(r'(?<![\d+−-])([01]?\d|2[0-3])(?:[:：]|点)([0-5]\d)(?:分)?(?!\d)',text):
            spans.append((match.start(),match.end(),('clock',int(match[1]),int(match[2]))))
    separator='.' if lang=='id' else ','
    pattern=r'[0-9]+(?:'+re.escape(separator)+r'[0-9]{3})*'
    for match in re.finditer(pattern,text):
        if any(match.start()<end and start<match.end() for start,end,_ in spans):continue
        before=text[max(0,match.start()-1):match.start()]
        after=text[match.end():match.end()+1]
        raw=match.group().replace(separator,'')
        # Avoid signs, fractional/grouped values, times, ranges, IDs and ordinals.
        adjacent_number = (bool(before and before in '.,:/-' and match.start()>1 and text[match.start()-2].isdigit())
                           or bool(after and after in '.,:/-' and text[match.end()+1:match.end()+2].isdigit()))
        invalid=(before in ('-','+','−') or adjacent_number
                 or len(raw)>6 or (len(raw)>1 and raw.startswith('0'))
                 or (lang in {'id','tl','en'} and any(x and (x.isalpha() or x=='_') for x in (before,after))))
        if invalid or not aliases(int(raw),lang):
            unresolved=True
        else:spans.append((match.start(),match.end(),int(raw)))
    return spans,unresolved


def numeric_equivalence(reference, hypothesis, lang):
    if lang not in {'id','th','tl','zh','en'}:
        raise ValueError('Unsupported review language')
    texts=[unicodedata.normalize('NFC',s).casefold().translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙','0123456789'))
           for s in (reference,hypothesis)]
    if any('⟦' in s or '⟧' in s for s in texts):
        raise ValueError('Reserved diagnostic marker in input')
    parsed=[_digits(s,lang) for s in texts]
    values={n for spans,_ in parsed for _,_,n in spans}
    forms={form:n for n in values for form in aliases(n,lang)}
    variants=sorted(forms,key=lambda s:(-len(s),s))
    expressions=[]
    for form in variants:
        pattern=re.escape(form).replace(r'\ ',r'[\s-]*').replace(r'\-',r'[\s-]*')
        if isinstance(forms[form],tuple):pass  # A complete clock expression already ends at minutes.
        elif lang in {'id','en','tl'}:pattern=r'(?<!\w)'+pattern+r'(?!\w)'
        elif lang=='th':
            pattern+=r'(?=$|\s|[.,!?;:]|ปี|วัน|คน|บาท|ครั้ง|นัด|เดือน|ชั่วโมง|นาที|วินาที|เมตร|กิโล|เปอร์เซ็นต์|ตัว)'
        else:pattern+=r'(?=$|\s|[。，！？.,!?;:]|年|天|人|个|次|月|日|小时|分钟|秒|米|岁|只|条|块|颗|公里|公斤)'
        expressions.append((re.compile(pattern),forms[form]))
    normalized=[]
    for text,(digits,_) in zip(texts,parsed):
        spans=list(digits)
        for pattern,n in expressions:
            for match in pattern.finditer(text):
                if not any(match.start()<end and start<match.end() for start,end,_ in spans):
                    spans.append((match.start(),match.end(),n))
        offset=0
        result=''
        for start,end,n in sorted(spans):
            result+=_norm(text[offset:start])+f'⟦{n}⟧'
            offset=end
        normalized.append(result+_norm(text[offset:]))
    unresolved=any(pending for _,pending in parsed)
    return {'numeric_only_difference':bool(values) and not unresolved
                and _norm(reference)!=_norm(hypothesis) and normalized[0]==normalized[1],
            'reference_view':normalized[0], 'hypothesis_view':normalized[1],
            'unresolved_numeric_format':unresolved,
            'policy':'Only exact integer spelling equivalence is ignored; mixed errors still need review.'}
